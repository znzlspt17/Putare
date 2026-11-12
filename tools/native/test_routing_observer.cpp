// 2026-11-09: 실제 callback에 합성 텐서를 주어 경량/기존 계측 형식과 복사 범위를 검증한다.
#define main putare_engine_entry_for_test
#include "expert_research.cpp"
#undef main

#include <cassert>

static int tensor_reads = 0;
extern "C" size_t ggml_nbytes(const ggml_tensor * tensor) {
    return tensor->nb[3] * static_cast<size_t>(tensor->ne[3]);
}
extern "C" void ggml_backend_tensor_get(const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ++tensor_reads;
    std::memcpy(data, static_cast<const char *>(tensor->data) + offset, size);
}
extern "C" void ggml_backend_tensor_set(ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    std::memcpy(static_cast<char *>(tensor->data) + offset, data, size);
}

static ggml_tensor tensor(const char * name, ggml_type type, void * data,
                          int64_t n0, int64_t n1, int64_t n2) {
    ggml_tensor result{};
    std::snprintf(result.name, sizeof(result.name), "%s", name);
    result.type=type; result.data=data;
    result.ne[0]=n0; result.ne[1]=n1; result.ne[2]=n2; result.ne[3]=1;
    result.nb[0]=4; result.nb[1]=4*n0; result.nb[2]=result.nb[1]*n1; result.nb[3]=result.nb[2]*n2;
    return result;
}

int main(int argc, char ** argv) {
    assert(argc==2);
    const char * path=argv[1];
    int32_t ids[]={1,3,2,0}; float gates[]={.75f,.25f,.4f,.6f};
    float weighted[]={3.f,4.f,0.f,2.f,6.f,8.f,0.f,5.f};
    auto topk=tensor("ffn_moe_topk-0",GGML_TYPE_I32,ids,2,2,1);
    auto norm=tensor("ffn_moe_weights_norm-0",GGML_TYPE_F32,gates,2,2,1);
    auto norms3=tensor("ffn_moe_weights_norm-0",GGML_TYPE_F32,gates,1,2,2);
    auto expert=tensor("ffn_moe_weighted-0",GGML_TYPE_F32,weighted,2,2,2);
    std::ofstream output(path);
    observer state; state.output=&output; state.enabled=true; state.instrumentation_level="routing";
    state.top_k=2; state.position=5;
    assert(callback(&topk,true,&state));
    assert(callback(&norm,true,&state));
    assert(!callback(&expert,true,&state));
    assert(callback(&topk,false,&state));
    assert(callback(&norm,false,&state));
    assert(tensor_reads==2);
    assert(state.records==2 && state.offsets.at(0)==2 && state.profiled_layers.count(0)==1);
    output.flush();
    std::ifstream input(path); std::string line;
    assert(std::getline(input,line)); auto first=json::parse(line);
    assert(first.at("token_index")==5 && first.at("expert_ids")==json::array({1,3}));
    assert(first.at("weights")==json::array({.75,.25}) && !first.contains("weighted_output_norms"));
    assert(std::getline(input,line)); auto second=json::parse(line);
    assert(second.at("token_index")==6 && second.at("expert_ids")==json::array({2,0}));
    assert(!std::getline(input,line));
    input.close(); output.close();

    // 두 가지 정규화 텐서 형식과 잘못된 id/가중치 쌍을 검증한다.
    output.open(path,std::ios::trunc); state=observer{}; state.output=&output; state.enabled=true;
    state.instrumentation_level="routing"; state.top_k=2;
    assert(callback(&topk,false,&state)); assert(callback(&norms3,false,&state));
    assert(state.records==2 && state.failure.empty());
    assert(!callback(&norms3,false,&state));
    assert(!state.failure.empty());
    output.close();

    output.open(path,std::ios::trunc); state=observer{}; state.output=&output; state.enabled=true;
    state.instrumentation_level="routing"; state.top_k=2;
    int32_t duplicate_ids[]={1,1,2,0};
    auto duplicate=tensor("ffn_moe_topk-0",GGML_TYPE_I32,duplicate_ids,2,2,1);
    assert(callback(&duplicate,false,&state));
    assert(!callback(&norm,false,&state) && !state.failure.empty());
    output.close();

    output.open(path,std::ios::trunc); state=observer{}; state.output=&output; state.enabled=true;
    state.instrumentation_level="routing"; state.top_k=2;
    float invalid_gates[]={.75f,.75f,.4f,.6f};
    auto invalid=tensor("ffn_moe_weights_norm-0",GGML_TYPE_F32,invalid_gates,2,2,1);
    assert(callback(&topk,false,&state));
    assert(!callback(&invalid,false,&state) && !state.failure.empty());
    output.close();

    // 기존 full은 weighted tensor를 관측하고 norm을 그대로 출력한다.
    output.open(path,std::ios::trunc); state=observer{}; state.output=&output; state.enabled=true; state.top_k=2;
    assert(callback(&expert,true,&state));
    assert(callback(&topk,false,&state)); assert(callback(&norm,false,&state)); assert(callback(&expert,false,&state));
    output.flush(); input.open(path); assert(std::getline(input,line)); first=json::parse(line);
    assert(first.at("weighted_output_norms")==json::array({5.0,2.0}));
    input.close(); output.close();
    return 0;
}
