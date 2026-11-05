// 2026-11-05: 원본 GGUF를 유지하며 scheduler의 실제 중간 텐서를 관측·제한한다.
#include "llama.h"
#include "ggml-backend.h"
#include "chat.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
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
    const std::string kind = name.substr(0, dash);
    const bool mask = kind == "ffn_moe_logits" && !state.allow.empty();
    const bool capture = state.enabled && (kind == "ffn_moe_topk" || kind == "ffn_moe_weights_norm" || kind == "ffn_moe_weighted");
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
            for (int64_t token=0; token<tensor->ne[1]; ++token)
                for (int64_t k=0; k<tensor->ne[0]; ++k)
                    gates.push_back(value_at<float>(bytes, token*tensor->nb[1] + k*tensor->nb[0]));
        } else {
            if (tensor->type != GGML_TYPE_F32) throw std::runtime_error("weighted output is not F32");
            const int64_t k_count=tensor->ne[1], tokens=tensor->ne[2];
            const auto & ids=state.ids.at(layer); const auto & gates=state.gates.at(layer);
            if (ids.size()!=static_cast<size_t>(k_count*tokens) || gates.size()!=ids.size()) throw std::runtime_error("routing vector order/shape mismatch");
            for (int64_t token=0; token<tokens; ++token) {
                if (++state.records > state.max_records) throw std::runtime_error("routing record limit exceeded");
                json selected=json::array(), weights=json::array(), norms=json::array();
                for (int64_t k=0; k<k_count; ++k) {
                    double square=0;
                    for (int64_t h=0; h<tensor->ne[0]; ++h) {
                        const float value=value_at<float>(bytes, h*tensor->nb[0]+k*tensor->nb[1]+token*tensor->nb[2]);
                        if (!std::isfinite(value)) throw std::runtime_error("nonfinite expert output");
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
        state.failure=error.what(); return false;
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
    }
    llama_batch_ext_set_output_logits(batch,static_cast<int>(tokens.size()-1),true);
    if(llama_process(context,LLAMA_PROCESS_TYPE_DECODE,batch)!=0) throw std::runtime_error("llama_process failed: "+state.failure);
    llama_synchronize(context);
    if(!state.failure.empty()) throw std::runtime_error(state.failure);
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
    try {
        if(argc!=5 || std::string(argv[1])!="--request" || std::string(argv[3])!="--output") throw std::runtime_error("usage: --request request.json --output result.jsonl");
        std::ifstream input(argv[2]); if(!input) throw std::runtime_error("request not readable");
        const json request=json::parse(input);
        output.open(argv[4],std::ios::out|std::ios::trunc); if(!output) throw std::runtime_error("output not writable");
        const std::string mode=request.value("mode",std::string("generate"));
        if(mode!="generate" && mode!="profile" && mode!="logits" && mode!="reference_loss") throw std::runtime_error("unsupported mode");
        const int limit=request.value("max_tokens",256), n_ctx=request.value("context_size",8192);
        if(limit<0 || n_ctx<=0 || limit>n_ctx) throw std::runtime_error("invalid token/context limit");
        observer state; state.output=&output; state.enabled=request.value("instrumentation",mode=="profile");
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
        llama_backend_init();
        const auto load_start=clock_type::now();
        auto model_params=llama_model_default_params(); model_params.n_gpu_layers=request.value("n_gpu_layers",0);
        model=llama_model_load_from_file(request.at("model").get<std::string>().c_str(),model_params);
        if(!model) throw std::runtime_error("model load failed");
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
            common_chat_msg message; message.role="user"; message.content=prompt; chat.messages.push_back(message);
            chat.enable_thinking=request.value("enable_thinking",false);
            chat.now=std::chrono::system_clock::time_point(std::chrono::seconds(0));
            prompt=common_chat_templates_apply(templates.get(),chat).prompt.str();
        }
        auto tokens=tokenize(vocab,prompt,true);
        auto reference=mode=="reference_loss" ? tokenize(vocab,request.at("reference").get<std::string>(),false) : std::vector<llama_token>();
        const int needed=static_cast<int>(tokens.size())+(mode=="reference_loss"?static_cast<int>(reference.size()):limit);
        if(tokens.empty() || needed>n_ctx) throw std::runtime_error("prompt/output exceeds context; no silent truncation");
        auto params=llama_context_default_params(); params.n_ctx=n_ctx; params.n_batch=64; params.n_ubatch=64; params.no_perf=false;
        if(state.enabled || !state.allow.empty()) { params.cb_eval=callback; params.cb_eval_user_data=&state; }
        context=llama_init_from_model(model,params); if(!context) throw std::runtime_error("context creation failed");
        batch=llama_batch_ext_init(context);
        const double load_seconds=elapsed(load_start);
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
                const auto token=static_cast<llama_token>(std::max_element(logits,logits+vocabulary)-logits);
                if(i==0) ttft=prefill_seconds+elapsed(decode_start);
                if(llama_vocab_is_eog(vocab,token)) { stop_reason="eog"; break; }
                generated.push_back(token); text+=piece(vocab,token);
                if(i+1<limit) process(context,batch,{token},position++,state);
            }
        }
        const double decode_seconds=elapsed(decode_start);
        const size_t expected_layers=static_cast<size_t>(llama_model_n_layer(model));
        if(!state.allow.empty() && state.masked_layers.size()!=expected_layers) throw std::runtime_error("mask callback did not observe every layer; graph unsupported");
        if(state.enabled && state.profiled_layers.size()!=expected_layers) throw std::runtime_error("profiling callbacks did not observe every layer; graph unsupported");
        if(mode=="logits") {
            for(const auto & value:final_logits) if(!std::isfinite(value.get<float>())) throw std::runtime_error("nonfinite final logits");
        }
        json result={{"type","result"},{"runtime_commit",RESEARCH_LLAMA_COMMIT},{"mode",mode},
            {"text",text},{"token_ids",generated},{"logits",final_logits},
            {"reference_nll",reference.empty()?json(nullptr):json(loss/reference.size())},{"reference_tokens",reference.size()},
            {"stop_reason",stop_reason},{"prompt_tokens",tokens.size()},{"seed",request.value("seed",42)},
            {"sampling",{{"temperature",0.0},{"strategy","greedy"}}},{"instrumentation",state.enabled},
            {"warmup",warmup},
            {"rendered_prompt",prompt},{"apply_chat_template",request.value("apply_chat_template",true)},
            {"enable_thinking",request.value("enable_thinking",false)},{"requested_n_gpu_layers",model_params.n_gpu_layers},
            {"routing_records",state.records},{"profiled_layers",state.profiled_layers},{"masked_layers",state.masked_layers},
            {"gpu_memory_bytes",nullptr},{"cpu_gpu_placement",nullptr},{"expert_transfer_bytes",nullptr},
            {"timings",{{"load_s",load_seconds},{"ttft_s",mode=="logits" || mode=="reference_loss"?json(nullptr):json(ttft)},
                {"prefill_tokens_s",tokens.size()/prefill_seconds},{"decode_tokens_s",generated.size()>1?json((generated.size()-1)/decode_seconds):json(nullptr)}}}};
        output << result.dump(-1,' ',false,json::error_handler_t::replace) << '\n'; output.flush();
        llama_batch_ext_free(batch); llama_free(context); llama_model_free(model); llama_backend_free(); return 0;
    } catch(const std::exception & error) {
        if(output) output << json({{"type","error"},{"message",error.what()}}).dump() << '\n';
        std::cerr << error.what() << '\n';
        if(batch) llama_batch_ext_free(batch); if(context) llama_free(context); if(model) llama_model_free(model);
        return 1;
    }
}
