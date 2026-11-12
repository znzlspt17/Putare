// 2026-11-05: 원본 GGUF를 유지하며 scheduler의 실제 중간 텐서를 관측·제한한다.
#include "llama.h"
#include "ggml-backend.h"
#include "chat.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;
using clock_type = std::chrono::steady_clock;
static double elapsed(clock_type::time_point start) {
    return std::chrono::duration<double>(clock_type::now() - start).count();
}
struct observer {
    std::ofstream * output = nullptr;
    bool enabled = false;
    std::string instrumentation_level = "full";
    size_t expected_layers = 0;
    bool all_token_logits = false;
    std::string phase = "prefill";
    int position = 0;
    int top_k = 8;
    std::map<int, std::set<int>> allow;
    std::map<int, std::vector<int32_t>> ids;
    std::map<int, std::vector<float>> gates;
    std::map<int, int> offsets;
    std::set<int> masked_layers;
    std::set<int> profiled_layers;
    std::string failure;
    size_t records = 0;
    size_t max_records = 1000000;
};

static std::vector<uint8_t> tensor_bytes(ggml_tensor * tensor) {
    std::vector<uint8_t> bytes(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
    return bytes;
}
template<typename T> static T value_at(const std::vector<uint8_t> & bytes, size_t offset) {
    if (offset + sizeof(T) > bytes.size()) throw std::runtime_error("tensor stride exceeds data");
    T value; std::memcpy(&value, bytes.data() + offset, sizeof(T)); return value;
}
static bool callback(ggml_tensor * tensor, bool ask, void * user) {
    auto & state = *static_cast<observer *>(user);
    const std::string name(tensor->name);
    const auto dash = name.rfind('-');
    if (dash == std::string::npos) return false;
    // 2026-11-06: 하위 VIEW의 상속 이름에 붙은 " (view)"를 원본 계측 노드로 오인하지 않는다.
    const std::string suffix = name.substr(dash + 1);
    if (suffix.empty() || !std::all_of(suffix.begin(), suffix.end(), [](char c) { return c >= '0' && c <= '9'; })) return false;
    const std::string kind = name.substr(0, dash);
    const bool mask = kind == "ffn_moe_logits" && !state.allow.empty();
    const bool capture = state.enabled && (kind == "ffn_moe_topk" || kind == "ffn_moe_weights_norm" ||
        (state.instrumentation_level == "full" && kind == "ffn_moe_weighted"));
    if (ask) return mask || capture;
    if (!mask && !capture) return true;
    try {
        const int layer = std::stoi(name.substr(dash + 1));
        if (mask) {
            if (tensor->type != GGML_TYPE_F32) throw std::runtime_error("router logits are not F32");
            const auto found = state.allow.find(layer);
            if (found == state.allow.end()) throw std::runtime_error("missing layer allowlist");
            for (const int id : found->second) if (id < 0 || id >= tensor->ne[0]) throw std::runtime_error("expert id out of range");
            if (found->second.size() < static_cast<size_t>(state.top_k)) throw std::runtime_error("allowlist smaller than top-k");
            auto bytes = tensor_bytes(tensor);
            const float excluded = -std::numeric_limits<float>::infinity();
            for (int64_t row = 0; row < tensor->ne[1]; ++row) {
                for (int64_t expert = 0; expert < tensor->ne[0]; ++expert) {
                    if (!found->second.count(static_cast<int>(expert))) {
                        std::memcpy(bytes.data() + row*tensor->nb[1] + expert*tensor->nb[0], &excluded, sizeof(float));
                    }
                }
            }
            // ask=false는 해당 노드 계산 후, 소비 노드 실행 전이다. Top-K 이전 logits만 변경한다.
            ggml_backend_tensor_set(tensor, bytes.data(), 0, bytes.size());
            state.masked_layers.insert(layer);
        }
        if (!capture) return true;
        const auto bytes = tensor_bytes(tensor);
        if (kind == "ffn_moe_topk") {
            if (tensor->type != GGML_TYPE_I32) throw std::runtime_error("top-k ids are not I32");
            auto & ids = state.ids[layer]; ids.clear();
            for (int64_t token=0; token<tensor->ne[1]; ++token)
                for (int64_t k=0; k<tensor->ne[0]; ++k)
                    ids.push_back(value_at<int32_t>(bytes, token*tensor->nb[1] + k*tensor->nb[0]));
        } else if (kind == "ffn_moe_weights_norm") {
            if (tensor->type != GGML_TYPE_F32) throw std::runtime_error("normalized gates are not F32");
            auto & gates = state.gates[layer]; gates.clear();
            // 2026-11-06: ggml reshape는 동일 텐서 이름을 공유할 수 있어 [K,T]와 [1,K,T]를 모두 보존한다.
            if (tensor->ne[0] == 1 && tensor->ne[1] == state.top_k) {
                for (int64_t token=0; token<tensor->ne[2]; ++token)
                    for (int64_t k=0; k<tensor->ne[1]; ++k)
                        gates.push_back(value_at<float>(bytes, token*tensor->nb[2] + k*tensor->nb[1]));
            } else {
                for (int64_t token=0; token<tensor->ne[1]; ++token)
                    for (int64_t k=0; k<tensor->ne[0]; ++k)
                        gates.push_back(value_at<float>(bytes, token*tensor->nb[1] + k*tensor->nb[0]));
            }
            if (state.instrumentation_level == "routing") {
                // 2026-11-09: 정규화 가중치와 선택 ID만 전송해 큰 expert 출력 복사와 norm 계산을 피한다.
                const auto found = state.ids.find(layer);
                if (found == state.ids.end() || found->second.size() != gates.size() ||
                    gates.empty() || gates.size() % static_cast<size_t>(state.top_k) != 0)
                    throw std::runtime_error("routing ids/gates shape or ordering mismatch");
                const auto & ids = found->second;
                const size_t tokens = gates.size() / static_cast<size_t>(state.top_k);
                for (size_t token=0; token<tokens; ++token) {
                    if (++state.records > state.max_records) throw std::runtime_error("routing record limit exceeded");
                    json selected=json::array(), weights=json::array();
                    std::set<int32_t> unique;
                    double total=0;
                    for (int k=0; k<state.top_k; ++k) {
                        const size_t index=token*static_cast<size_t>(state.top_k)+static_cast<size_t>(k);
                        const int32_t id=ids[index]; const float weight=gates[index];
                        if(id<0 || !unique.insert(id).second || !std::isfinite(weight) || weight<0)
                            throw std::runtime_error("invalid routing id or normalized weight");
                        selected.push_back(id); weights.push_back(weight); total+=weight;
                    }
                    if(std::abs(total-1.0)>1e-4) throw std::runtime_error("normalized routing weights do not sum to one");
                    *state.output << json({{"type","routing"},{"phase",state.phase},{"layer",layer},
                        {"token_index",state.position+state.offsets[layer]+static_cast<int>(token)},
                        {"expert_ids",selected},{"weights",weights}}).dump() << '\n';
                }
                state.offsets[layer] += static_cast<int>(tokens);
                state.profiled_layers.insert(layer);
                state.ids.erase(layer); state.gates.erase(layer);
            }
        } else {
            if (tensor->type != GGML_TYPE_F32) throw std::runtime_error("weighted output is not F32");
            const int64_t k_count=tensor->ne[1], tokens=tensor->ne[2];
            const auto & ids=state.ids.at(layer); const auto & gates=state.gates.at(layer);
            if (ids.size()!=static_cast<size_t>(k_count*tokens) || gates.size()!=ids.size()) throw std::runtime_error("routing vector order/shape mismatch layer=" + std::to_string(layer) + " ids=" + std::to_string(ids.size()) + " gates=" + std::to_string(gates.size()) + " weighted=" + std::to_string(tensor->ne[0]) + "," + std::to_string(k_count) + "," + std::to_string(tokens));
            for (int64_t token=0; token<tokens; ++token) {
                if (++state.records > state.max_records) throw std::runtime_error("routing record limit exceeded");
                json selected=json::array(), weights=json::array(), norms=json::array();
                for (int64_t k=0; k<k_count; ++k) {
                    double square=0;
                    for (int64_t h=0; h<tensor->ne[0]; ++h) {
                        const float value=value_at<float>(bytes, h*tensor->nb[0]+k*tensor->nb[1]+token*tensor->nb[2]);
                        if (!std::isfinite(value)) throw std::runtime_error("nonfinite expert output layer=" + std::to_string(layer) + " token=" + std::to_string(token) + " k=" + std::to_string(k) + " h=" + std::to_string(h));
                        square += static_cast<double>(value)*value;
                    }
                    selected.push_back(ids[token*k_count+k]); weights.push_back(gates[token*k_count+k]); norms.push_back(std::sqrt(square));
                }
                *state.output << json({{"type","routing"},{"phase",state.phase},{"layer",layer},
                    {"token_index",state.position+state.offsets[layer]+token},{"expert_ids",selected},
                    {"weights",weights},{"weighted_output_norms",norms}}).dump() << '\n';
            }
            state.offsets[layer] += static_cast<int>(tokens);
            state.profiled_layers.insert(layer);
        }
        return true;
    } catch (const std::exception & error) {
        if (state.failure.empty()) state.failure=error.what(); return false;
    }
}

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text, bool special) {
    const int count=-llama_tokenize(vocab,text.data(),static_cast<int>(text.size()),nullptr,0,special,true);
    if(count<0) throw std::runtime_error("token count failed");
    std::vector<llama_token> tokens(count);
    if(llama_tokenize(vocab,text.data(),static_cast<int>(text.size()),tokens.data(),count,special,true)<0)
        throw std::runtime_error("tokenization failed");
    return tokens;
}
static std::string piece(const llama_vocab * vocab,llama_token token) {
    std::vector<char> bytes(256);
    int size=llama_token_to_piece(vocab,token,bytes.data(),static_cast<int>(bytes.size()),0,true);
    if(size<0) { bytes.resize(-size); size=llama_token_to_piece(vocab,token,bytes.data(),static_cast<int>(bytes.size()),0,true); }
    if(size<0) throw std::runtime_error("token piece failed");
    return std::string(bytes.data(),size);
}
static void process(llama_context * context,llama_batch_ext * batch,const std::vector<llama_token> & tokens,int position,observer & state) {
    state.position=position; state.offsets.clear();
    llama_batch_ext_clear(batch);
    for(size_t i=0;i<tokens.size();++i) {
        const int index=llama_batch_ext_add_token(batch,0,tokens[i]);
        const llama_pos pos=position+static_cast<int>(i);
        llama_batch_ext_set_pos(batch,index,&pos);
        if(state.all_token_logits) llama_batch_ext_set_output_logits(batch,index,true);
    }
    llama_batch_ext_set_output_logits(batch,static_cast<int>(tokens.size()-1),true);
    if(llama_process(context,LLAMA_PROCESS_TYPE_DECODE,batch)!=0) throw std::runtime_error("llama_process failed: "+state.failure);
    llama_synchronize(context);
    if(!state.failure.empty()) throw std::runtime_error(state.failure);
    if(state.enabled && state.instrumentation_level=="routing") {
        // 2026-11-09: 일부 토큰 또는 계층에서 callback이 누락되면 조용히 결과를 쓰지 않는다.
        if(state.offsets.size()!=state.expected_layers) throw std::runtime_error("routing callback missed a layer");
        for(size_t layer=0;layer<state.expected_layers;++layer) {
            auto found=state.offsets.find(static_cast<int>(layer));
            if(found==state.offsets.end() || found->second!=static_cast<int>(tokens.size()))
                throw std::runtime_error("routing callback missed or duplicated tokens");
        }
    }
}
static double nll(const float * logits,int count,int target) {
    if(target<0 || target>=count) throw std::runtime_error("reference token outside vocabulary");
    for(int i=0;i<count;++i) if(!std::isfinite(logits[i])) throw std::runtime_error("nonfinite reference logits");
    const float maximum=*std::max_element(logits,logits+count);
    double total=0; for(int i=0;i<count;++i) total+=std::exp(static_cast<double>(logits[i])-maximum);
    return maximum+std::log(total)-logits[target];
}

int main(int argc,char ** argv) {
    std::ofstream output;
    llama_model * model=nullptr; llama_context * context=nullptr; llama_batch_ext * batch=nullptr;
    llama_sampler * sampler=nullptr;
    // 2026-11-07: callback이 보관하는 주소를 세션 전체에서 고정한다.
    observer state;
    json session_config;
    const bool session=argc==2 && std::string(argv[1])=="--session";
    bool backend_started=false;
    try {
        if(!session && (argc!=5 || std::string(argv[1])!="--request" || std::string(argv[3])!="--output")) throw std::runtime_error("usage: --request request.json --output result.jsonl | --session");
        while(true) {
        std::string request_path, output_path;
        if(session) {
            std::string line;
            if(!std::getline(std::cin,line)) break;
            const json command=json::parse(line);
            request_path=command.at("request").get<std::string>();
            output_path=command.at("output").get<std::string>();
        } else { request_path=argv[2]; output_path=argv[4]; }
        std::ifstream input(request_path); if(!input) throw std::runtime_error("request not readable");
        const json request=json::parse(input);
        output.open(output_path,std::ios::out|std::ios::trunc); if(!output) throw std::runtime_error("output not writable");
        const std::string mode=request.value("mode",std::string("generate"));
        if(mode!="generate" && mode!="profile" && mode!="logits" && mode!="reference_loss" && mode!="tokenize") throw std::runtime_error("unsupported mode");
        const int limit=request.value("max_tokens",256), n_ctx=request.value("context_size",8192);
        if(limit<0 || n_ctx<=0 || limit>n_ctx) throw std::runtime_error("invalid token/context limit");
        // 2026-11-06: 기존 연구 호출은 greedy를 보존하고 명시된 경우에만 모델 권장 샘플링을 사용한다.
        const auto sampling=request.value("sampling",json::object());
        const float temperature=sampling.value("temperature",0.0f), top_p=sampling.value("top_p",1.0f);
        const int sample_top_k=sampling.value("top_k",0);
        if(!std::isfinite(temperature) || temperature<0 || !std::isfinite(top_p) || top_p<=0 || top_p>1 || sample_top_k<0)
            throw std::runtime_error("invalid sampling");
        const std::string cache_k=request.value("cache_type_k",std::string("f16"));
        const std::string cache_v=request.value("cache_type_v",std::string("f16"));
        if((cache_k!="f16" && cache_k!="q8_0") || (cache_v!="f16" && cache_v!="q8_0"))
            throw std::runtime_error("unsupported cache type");
        state=observer{}; state.output=&output; state.enabled=request.value("instrumentation",mode=="profile");
        state.instrumentation_level=request.value("instrumentation_level",std::string("full"));
        if(state.instrumentation_level!="full" && state.instrumentation_level!="routing")
            throw std::runtime_error("unsupported instrumentation level");
        const bool fusion_enabled=request.value("fusion_enabled",false);
        state.all_token_logits=request.value("all_token_logits",false);
        if(state.enabled && state.instrumentation_level=="routing" && !state.all_token_logits)
            throw std::runtime_error("routing instrumentation requires all_token_logits");
        const bool warmup=request.value("warmup",false);
        if(warmup && (state.enabled || mode!="generate")) throw std::runtime_error("warmup requires uninstrumented generate mode");
        state.max_records=request.value("max_routing_records",size_t(1000000));
        if(request.contains("allowlist") && !request["allowlist"].is_null()) {
            if(!request["allowlist"].is_object() || request["allowlist"].empty()) throw std::runtime_error("allowlist must be a nonempty object or null");
            for(auto it=request["allowlist"].begin();it!=request["allowlist"].end();++it) {
                const int layer=std::stoi(it.key());
                for(const int id:it.value()) if(!state.allow[layer].insert(id).second) throw std::runtime_error("duplicate expert id");
            }
        }
        const json fixed={{"model",request.at("model")},{"n_gpu_layers",request.value("n_gpu_layers",0)},
            {"vocab_only",mode=="tokenize"},
            {"context_size",n_ctx},{"cache_k",cache_k},{"cache_v",cache_v},
            {"flash_attention",request.value("flash_attention",false)},
            {"offload_kqv",request.value("offload_kqv",true)},
            {"instrumentation",state.enabled},{"instrumentation_level",state.instrumentation_level},
            {"fusion_enabled",fusion_enabled},{"allowlist",request.value("allowlist",json(nullptr))}};
        if(model && fixed!=session_config) throw std::runtime_error("session model/context configuration changed");
        if(!backend_started) {
            // 2026-11-09: CUDA가 최초 조회 때 env를 캐시하므로 backend 초기화 전에 세션 정책을 고정한다.
#ifdef _WIN32
            if(_putenv_s("GGML_CUDA_DISABLE_FUSION",fusion_enabled ? "0" : "1")!=0)
#else
            if(setenv("GGML_CUDA_DISABLE_FUSION",fusion_enabled ? "0" : "1",1)!=0)
#endif
                throw std::runtime_error("failed to set CUDA fusion policy");
            llama_backend_init(); backend_started=true;
        }
        const auto load_start=clock_type::now();
        auto model_params=llama_model_default_params(); model_params.n_gpu_layers=request.value("n_gpu_layers",0);
        // 2026-11-11: 비교 사전 점검은 가중치 적재나 평가 추론 없이 실제 chat template과 vocab만 사용한다.
        model_params.vocab_only=(mode=="tokenize");
        const bool first=!model;
        if(first) { model=llama_model_load_from_file(request.at("model").get<std::string>().c_str(),model_params); session_config=fixed; }
        if(!model) throw std::runtime_error("model load failed");
        state.expected_layers=static_cast<size_t>(llama_model_n_layer(model));
        char arch[128];
        if(llama_model_meta_val_str(model,"general.architecture",arch,sizeof(arch))<=0) throw std::runtime_error("missing model architecture");
        if(std::string(arch)!="gemma4") throw std::runtime_error("this observer supports gemma4 only");
        char top_k[32];
        if(llama_model_meta_val_str(model,"gemma4.expert_used_count",top_k,sizeof(top_k))<=0) throw std::runtime_error("missing expert_used_count");
        state.top_k=std::stoi(top_k);
        if(state.top_k<=0) throw std::runtime_error("invalid original top-k");
        if(!state.allow.empty()) {
            if(state.allow.size()!=static_cast<size_t>(llama_model_n_layer(model))) throw std::runtime_error("allowlist must cover every layer");
            for(int i=0;i<llama_model_n_layer(model);++i) {
                if(!state.allow.count(i)) throw std::runtime_error("missing layer allowlist");
                // 2026-11-05: 단일 계층 ablation은 계층별 유지 수가 다르다. 균일 M은 물리 export만의 제약이다.
                if(state.allow[i].size()<static_cast<size_t>(state.top_k)) throw std::runtime_error("allowlist smaller than top-k");
            }
        }
        const llama_vocab * vocab=llama_model_get_vocab(model);
        std::string prompt=request.at("prompt").get<std::string>();
        if(request.value("apply_chat_template",true)) {
            auto templates=common_chat_templates_init(model,"");
            common_chat_templates_inputs chat;
            if(request.contains("system_prompt") && !request.at("system_prompt").get<std::string>().empty()) {
                common_chat_msg system; system.role="system"; system.content=request.at("system_prompt").get<std::string>(); chat.messages.push_back(system);
            }
            common_chat_msg message; message.role="user"; message.content=prompt; chat.messages.push_back(message);
            chat.enable_thinking=request.value("enable_thinking",false);
            chat.now=std::chrono::system_clock::time_point(std::chrono::seconds(0));
            prompt=common_chat_templates_apply(templates.get(),chat).prompt.str();
        }
        auto tokens=tokenize(vocab,prompt,true);
        if(mode=="tokenize") {
            if(tokens.empty() || tokens.size()>=static_cast<size_t>(n_ctx-1)) throw std::runtime_error("prompt exceeds context");
            output << json({{"type","result"},{"rendered_prompt",prompt},{"tokens",tokens}}).dump() << '\n';
            output.close();
            if(!output) throw std::runtime_error("tokenization output failed");
            if(!session) break;
            std::cout << json({{"status","ok"}}).dump() << std::endl;
            continue;
        }
        if(request.contains("expected_prompt_tokens") && request.at("expected_prompt_tokens")!=json(tokens))
            throw std::runtime_error("prompt tokens differ from verified preflight");
        auto reference=mode=="reference_loss" ? tokenize(vocab,request.at("reference").get<std::string>(),false) : std::vector<llama_token>();
        const int needed=static_cast<int>(tokens.size())+(mode=="reference_loss"?static_cast<int>(reference.size()):limit);
        if(tokens.empty() || needed>n_ctx) throw std::runtime_error("prompt/output exceeds context; no silent truncation");
        auto params=llama_context_default_params(); params.n_ctx=n_ctx; params.n_batch=64; params.n_ubatch=64; params.no_perf=false;
        params.type_k=cache_k=="q8_0" ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
        params.type_v=cache_v=="q8_0" ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
        if(request.contains("flash_attention")) params.flash_attn_type=request.at("flash_attention").get<bool>() ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        if(request.contains("offload_kqv")) params.offload_kqv=request.at("offload_kqv").get<bool>();
        if(state.enabled || !state.allow.empty()) { params.cb_eval=callback; params.cb_eval_user_data=&state; }
        if(first) { context=llama_init_from_model(model,params); if(!context) throw std::runtime_error("context creation failed"); }
        else {
            // 기존 문제의 KV와 callback 통계를 지우되 모델과 할당된 context는 유지한다.
            llama_synchronize(context);
            auto memory=llama_get_memory(context);
            if(!memory) throw std::runtime_error("session requires resettable KV memory");
            llama_memory_clear(memory,true);
        }
        if(temperature>0) {
            sampler=llama_sampler_chain_init(llama_sampler_chain_default_params());
            if(!sampler) throw std::runtime_error("sampling initialization failed");
            if(sample_top_k>0) llama_sampler_chain_add(sampler,llama_sampler_init_top_k(sample_top_k));
            if(top_p<1) llama_sampler_chain_add(sampler,llama_sampler_init_top_p(top_p,1));
            llama_sampler_chain_add(sampler,llama_sampler_init_temp(temperature));
            llama_sampler_chain_add(sampler,llama_sampler_init_dist(request.value("seed",42u)));
        }
        if(first) batch=llama_batch_ext_init(context);
        const double load_seconds=first ? elapsed(load_start) : 0.0;
        if(warmup) {
            // 2026-11-05: 같은 프로세스에서 kernel/prompt 경로를 준비한 후 KV를 지워 실제 측정 문맥을 분리한다.
            int warm_position=0;
            for(size_t offset=0;offset<tokens.size();offset+=64) {
                std::vector<llama_token> chunk(tokens.begin()+offset,tokens.begin()+std::min(tokens.size(),offset+64));
                process(context,batch,chunk,warm_position,state); warm_position+=static_cast<int>(chunk.size());
            }
            if(warm_position<n_ctx) {
                const float * logits=llama_get_logits_ith(context,-1);
                const int vocabulary=llama_vocab_n_tokens(vocab);
                const auto token=static_cast<llama_token>(std::max_element(logits,logits+vocabulary)-logits);
                process(context,batch,{token},warm_position,state);
            }
            llama_synchronize(context);
            auto memory=llama_get_memory(context);
            if(!memory) throw std::runtime_error("warmup requires resettable KV memory");
            llama_memory_clear(memory,true);
            state.masked_layers.clear(); state.offsets.clear();
        }
        const auto prefill_start=clock_type::now();
        int position=0;
        for(size_t offset=0;offset<tokens.size();offset+=64) {
            std::vector<llama_token> chunk(tokens.begin()+offset,tokens.begin()+std::min(tokens.size(),offset+64));
            process(context,batch,chunk,position,state); position+=static_cast<int>(chunk.size());
        }
        const double prefill_seconds=elapsed(prefill_start);
        const int vocabulary=llama_vocab_n_tokens(vocab);
        json final_logits=nullptr; if(mode=="logits") final_logits=std::vector<float>(llama_get_logits_ith(context,-1),llama_get_logits_ith(context,-1)+vocabulary);
        double loss=0, ttft=0; std::string text, stop_reason="length"; std::vector<llama_token> generated;
        const auto decode_start=clock_type::now();
        if(mode=="reference_loss") {
            if(reference.empty()) throw std::runtime_error("reference token sequence empty");
            // 각 reference token은 직전 문맥의 logits로 평가한다. prompt의 loss는 포함하지 않는다.
            state.phase="decode";
            for(size_t i=0;i<reference.size();++i) {
                loss+=nll(llama_get_logits_ith(context,-1),vocabulary,reference[i]);
                if(i+1<reference.size()) process(context,batch,{reference[i]},position++,state);
            }
            stop_reason="reference_complete";
        } else if(mode!="logits") {
            state.phase="decode";
            for(int i=0;i<limit;++i) {
                const float * logits=llama_get_logits_ith(context,-1);
                const auto token=sampler ? llama_sampler_sample(sampler,context,-1) : static_cast<llama_token>(std::max_element(logits,logits+vocabulary)-logits);
                if(sampler) llama_sampler_accept(sampler,token);
                if(i==0) ttft=prefill_seconds+elapsed(decode_start);
                if(llama_vocab_is_eog(vocab,token)) { stop_reason="eog"; break; }
                generated.push_back(token); text+=piece(vocab,token);
                if(i+1<limit) process(context,batch,{token},position++,state);
            }
        }
        const double decode_seconds=elapsed(decode_start);
        // 2026-11-11: 채널 경계가 토큰 경계와 일치할 때만 실제 출력 토큰을 분류한다.
        json channel_tokens={{"thinking",nullptr},{"final",nullptr},{"control",nullptr},
                             {"unclassified",generated.size()}};
        const std::string thought_marker="<|channel>thought\n", boundary="<channel|>";
        const size_t boundary_at=text.find(boundary);
        if(text.rfind(thought_marker,0)==0 && boundary_at!=std::string::npos) {
            std::vector<size_t> offsets={0};
            for(const auto token:generated) offsets.push_back(offsets.back()+piece(vocab,token).size());
            const size_t start=thought_marker.size(), end=boundary_at, final_start=end+boundary.size();
            auto a=std::find(offsets.begin(),offsets.end(),start);
            auto b=std::find(offsets.begin(),offsets.end(),end);
            auto c=std::find(offsets.begin(),offsets.end(),final_start);
            if(offsets.back()==text.size() && a!=offsets.end() && b!=offsets.end() && c!=offsets.end()) {
                const size_t ai=static_cast<size_t>(a-offsets.begin()), bi=static_cast<size_t>(b-offsets.begin());
                const size_t ci=static_cast<size_t>(c-offsets.begin());
                channel_tokens={{"thinking",bi-ai},{"final",generated.size()-ci},
                                {"control",ai+ci-bi},{"unclassified",0}};
            }
        }
        const size_t expected_layers=static_cast<size_t>(llama_model_n_layer(model));
        if(!state.allow.empty() && state.masked_layers.size()!=expected_layers) throw std::runtime_error("mask callback did not observe every layer; graph unsupported");
        if(state.enabled && state.profiled_layers.size()!=expected_layers) throw std::runtime_error("profiling callbacks did not observe every layer; graph unsupported");
        if(mode=="logits") {
            for(const auto & value:final_logits) if(!std::isfinite(value.get<float>())) throw std::runtime_error("nonfinite final logits");
        }
        json result={{"type","result"},{"runtime_commit",RESEARCH_LLAMA_COMMIT},{"gpu_fusion_disabled",!fusion_enabled},
            {"fusion_enabled",fusion_enabled},{"instrumentation_level",state.instrumentation_level},{"mode",mode},
            {"text",text},{"token_ids",generated},{"channel_tokens",channel_tokens},{"logits",final_logits},
            {"reference_nll",reference.empty()?json(nullptr):json(loss/reference.size())},{"reference_tokens",reference.size()},
            {"stop_reason",stop_reason},{"prompt_tokens",tokens.size()},{"seed",request.value("seed",42)},
            {"sampling",{{"temperature",temperature},{"top_p",top_p},{"top_k",sample_top_k},{"strategy",sampler?"sampled":"greedy"}}},{"instrumentation",state.enabled},
            {"cache_type_k",cache_k},{"cache_type_v",cache_v},{"flash_attention_requested",request.value("flash_attention",false)},
            {"offload_kqv_requested",params.offload_kqv},{"actual_context_size",llama_n_ctx(context)},
            {"processed_decode_tokens",position-static_cast<int>(tokens.size())},
            {"warmup",warmup},
            {"rendered_prompt",prompt},{"apply_chat_template",request.value("apply_chat_template",true)},
            {"enable_thinking",request.value("enable_thinking",false)},{"requested_n_gpu_layers",model_params.n_gpu_layers},
            {"routing_records",state.records},{"profiled_layers",state.profiled_layers},{"masked_layers",state.masked_layers},
            {"gpu_memory_bytes",nullptr},{"cpu_gpu_placement",nullptr},{"expert_transfer_bytes",nullptr},
            {"timings",{{"load_s",load_seconds},{"ttft_s",mode=="logits" || mode=="reference_loss"?json(nullptr):json(ttft)},
                {"prefill_s",prefill_seconds},{"decode_s",decode_seconds},
                {"prefill_tokens_s",tokens.size()/prefill_seconds},{"decode_tokens_s",generated.size()>1?json((generated.size()-1)/decode_seconds):json(nullptr)}}}};
        output << result.dump(-1,' ',false,json::error_handler_t::replace) << '\n'; output.flush();
        if(!output) throw std::runtime_error("result output failed");
        output.close();
        if(!output) throw std::runtime_error("result output close failed");
        if(sampler) { llama_sampler_free(sampler); sampler=nullptr; }
        if(!session) break;
        std::cout << json({{"status","ok"}}).dump() << std::endl;
        }
        if(batch) llama_batch_ext_free(batch); if(context) llama_free(context); if(model) llama_model_free(model); if(backend_started) llama_backend_free(); return 0;
    } catch(const std::exception & error) {
        if(output) output << json({{"type","error"},{"message",error.what()}}).dump() << '\n';
        std::cerr << error.what() << '\n';
        if(sampler) llama_sampler_free(sampler); if(batch) llama_batch_ext_free(batch); if(context) llama_free(context); if(model) llama_model_free(model); if(backend_started) llama_backend_free();
        return 1;
    }
}
