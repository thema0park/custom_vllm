# AI 개발 가이드: vLLM 커스텀 연산 최적화 프로젝트

## 📌 프로젝트 개요

vLLM 최신 버전에 커스텀 연산을 추가하고 성능을 최적화하는 프로젝트. CUDA 커널 개발, PyTorch 통합, vLLM 아키텍처 활용이 핵심입니다.

## 🏗️ 아키텍처 영역

### 주요 컴포넌트
- **CUDA 커널** (`src/custom_ops/`): 저수준 GPU 연산 최적화
- **PyTorch 바인딩** (`torch_binding.cpp`): CUDA ↔ Python 인터페이스
- **vLLM 통합** (`vllm_integration/`): Executor 및 Scheduler 후킹
- **벤치마킹 도구** (`src/benchmark/`): 성능 측정 및 분석 자동화

### 데이터 흐름
```
Input Tensor → CUDA Kernel → PyTorch Binding → vLLM Executor
              ↓                 ↓               ↓
          GPU Memory      Device Transfer   Batched Processing
```

## 🔧 개발 워크플로우

### 환경 설정
```bash
# 1. vLLM 개발 환경 설정
cd vllm
pip install -e ".[dev]"
python setup.py build_ext --inplace

# 2. 커스텀 연산 모듈 빌드
cd ../src/custom_ops
./build_cuda.sh  # or: python setup.py build_ext --inplace
```

### 테스트 및 검증
```bash
# 정확성 검증 (PyTorch baseline과 비교)
pytest tests/test_correctness.py -v

# 성능 프로파일링
python -m cProfile -s cumtime src/benchmark/benchmark_custom_op.py

# 통합 테스트 (vLLM 내에서 실행)
pytest tests/test_integration.py
```

### 빌드 및 배포
```bash
# 릴리스 빌드
python setup.py build_ext --inplace --release

# 벤치마크 실행
bash scripts/run_benchmarks.sh > results/benchmark_$(date +%Y%m%d).txt
```

## 💡 프로젝트 특화 패턴: Custom Attention 최적화

### CUDA Attention 커널 개발 (RTX 1660 Super / 2080 Ti 기준)

#### 1. 커널 설계 (GQA 기반)
- **타겟**: LLaMA 3.2의 Grouped Query Attention 구현
  - Query heads: 8, KV heads: 2 (그룹당 쿼리 4개)
  - Hidden dim: 128/head
- **스레드 블록 크기**: 32×8 (256 threads, 요청별 8개 토큰 처리)
  - 다른 옵션: 64×4 (256 threads, 더 높은 occupancy)
- **Shared Memory**: 128×8×4 bytes = 4KB (Query) + 4KB (Key) = 8KB
- **Occupancy 목표**: 50-75% (RTX 1660 S: 레지스터 압박 주의)

#### 2. 메모리 최적화 우선순위
1. **메모리 접근 패턴** (가장 중요 - 메모리 대역폭이 병목)
   - Coalesce global memory access: 연속 스레드가 연속 메모리 접근
   - Key/Value는 배치 내 공유 (Query는 스레드별 독립)
2. **Shared Memory 활용**
   - Query 벡터 캐싱 (레지스터 압력 감소)
   - Softmax 자체 (intermediate results)
3. **레지스터 사용 최소화**
   - 목표: thread당 128 레지스터 이하
   - 스파일 spillage 없음 (spillage = 8배 성능 저하)

#### 3. 스레드 배치 전략
```cuda
// Block configuration for RTX 1660 S
int block_size_x = 32;  // 쿼리 토큰 처리
int block_size_y = 8;   // 배치 인스턴스
int block_size = block_size_x * block_size_y;  // 256 threads

// Grid configuration
int grid_size = (seq_len + block_size_x - 1) / block_size_x * batch_size / block_size_y;
```

#### 4. 성능 보정 (RTX 1660 Super 특화)
- **메모리 대역폭**: 336 GB/s (주요 제약)
- **이론적 최대 throughput**: 336GB/s ÷ 16bytes/output = 21B outputs/s
- **배치 크기 영향**: 배치 ≤4 권장 (메모리 병목 동안 충분한 워프 보호)
- **Kernel launch overhead**: ≥100μs, 따라서 배치 작지 않으면 숨김

#### 5. Occupancy 계산
```cpp
// RTX 1660 S (sm_75): 96KB shared memory/SM, 32 warps/SM
// 커널 점유율 확인: nvidia-smi/nsys 또는 CUDA compute capability
int shared_mem_bytes = 8192;  // Query + Key cache
int registers_per_thread = 120;  // 추정
int occupancy = ceil(shared_mem_bytes / 96000) == 1 ? 
                min(32, 192000 / (256 * registers_per_thread)) : occupancy_limited;
// => 목표 occupancy 50%+ 달성
```

#### 6. 커널 네이밍 및 구성
```cuda
// custom_kernel.cu
__global__ void custom_attention_kernel(
    float *query,        // [batch, num_heads, seq_len, head_dim]
    float *key,          // [batch, num_kv_heads, seq_len, head_dim]
    float *value,        // [batch, num_kv_heads, seq_len, head_dim]
    float *output,       // [batch, num_heads, seq_len, head_dim]
    int seq_len,
    int head_dim,
    int num_heads,
    int num_kv_heads,
    float scale
) {
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    // ... 구현
}
```

**파일 위치**: `src/custom_ops/custom_attention_kernel.cu`

### PyTorch 바인딩 (Custom Attention)
- **C++ 래퍼**: `custom_attention_kernel.cuh` (extern 함수 + 헬퍼)
  ```cpp
  void launch_custom_attention(
      torch::Tensor query, torch::Tensor key, torch::Tensor value,
      torch::Tensor output, float scale
  );
  ```
- **pybind11 바인딩**: `torch_binding.cpp`
  ```cpp
  m.def("custom_attention", &custom_attention_forward,
        "Query"_a, "Key"_a, "Value"_a, "scale"_a);
  ```
- **타입 검증**: query, key, value의 dtype 일치성 확인 (FP32/FP16)
- **메모리 배치**: contiguous tensor 보장 (`.contiguous()`)

**파일 위치**: `src/custom_ops/torch_binding.cpp` (Attention 오버로드 추가)

### vLLM 통합: Custom Attention 후킹
- **통합점**: LLaMA 모델의 Attention 레이어
  - `vllm/model_executor/layers/attention.py` → `_execute_attention()` 메서드 오버라이드
  - vLLM의 AttentionMetadata 활용 (배치 정보, KVCache 관리)
- **KVCache 호환성**: vLLM의 KVCache 포맷 유지
  ```python
  # vLLM KVCache 구조 (batch, num_blocks, block_size, num_heads, head_dim)
  # 커스텀 커널이 이 포맷 직접 사용
  ```
- **배치 처리**: Paged Attention 메커니즘 활용 (vLLM scheduler 조율)
- **메모리 할당**: vLLM의 GPU allocator 사용 (메모리 누수 방지)

**파일 위치**: `src/vllm_integration/executor_patch.py` (LLaMA attention 후킹)

**예상 구현**:
```python
class CustomAttentionLLaMA(AttentionBase):
    def forward(self, query, key, value, attn_metadata):
        # vLLM KVCache 변환
        output = custom_ops.attention(
            query, key, value,
            scale=self.scale,
            seq_start_loc=attn_metadata.seq_start_loc
        )
        return output
```

### 성능 측정 (2배 달성 검증)
- **4가지 핵심 지표**:
  1. **Latency (ms)**: 단일 토큰 생성 시간
  2. **Throughput (tokens/sec)**: 초당 처리량
  3. **Memory (GB)**: Attention 레이어별 피크 메모리
  4. **Speedup (배수)**: PyTorch baseline 대비 성능 비율

- **측정 범위** (LLaMA 3.2 3B 기준):
  - Batch sizes: [1, 4, 8, 16, 32]
  - Sequence lengths: [128, 512, 2048, 8192]
  - 각 조합 10회 반복 (평균 및 편차 계산)

- **동기화 필수사항**:
  ```python
  torch.cuda.synchronize()  # 모든 측정 전후
  cuda_event_start = torch.cuda.Event(enable_timing=True)
  cuda_event_end = torch.cuda.Event(enable_timing=True)
  cuda_event_start.record()
  # ... 실행 ...
  cuda_event_end.record()
  elapsed_ms = cuda_event_start.elapsed_time(cuda_event_end)
  ```

- **결과 저장**: CSV 형식으로 자동 기록
  ```
  batch_size,seq_len,pytorch_latency_ms,custom_latency_ms,speedup_ratio
  ```

**파일 위치**: `src/benchmark/benchmark_attention.py` (Custom Attention 전용)

## 🔄 Cross-Component 통신

### CUDA ↔ PyTorch
```cpp
// custom_kernel.cu에서
__global__ void custom_kernel(float* input, float* output, int size) { ... }

// torch_binding.cpp에서 PyTorch tensor로 변환
at::Tensor custom_op_forward(at::Tensor input) {
    auto output = at::zeros_like(input);
    custom_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        input.numel()
    );
    return output;
}
```

### PyTorch ↔ vLLM
```python
# vllm_integration/executor_patch.py에서 후킹
class CustomExecutor(GPUExecutor):
    def execute_model(self, model_input):
        # ... 기존 로직 ...
        output = custom_op.forward(intermediate_tensor)
        # ... 남은 로직 ...
        return output
```

## 📋 핵심 파일 및 목적

| 파일 | 목적 | 주요 작업 |
|------|------|---------|
| `src/custom_ops/custom_kernel.cu` | CUDA 커널 구현 | GPU 연산 로직 |
| `src/custom_ops/torch_binding.cpp` | PyTorch 통합 | C++/Python 바인딩 |
| `src/vllm_integration/executor_patch.py` | vLLM 후킹 | 실행 파이프라인 통합 |
| `src/benchmark/benchmark_custom_op.py` | 성능 측정 자동화 | 벤치마크 데이터 수집 |
| `tests/test_correctness.py` | 정확성 검증 | baseline 대비 비교 |
| `docs/ARCHITECTURE.md` | 설계 문서 | 전체 흐름 설명 |

## 🎯 Common Tasks

### Custom Attention 커널 개발 & 최적화
1. **초기 구현**: `src/custom_ops/custom_attention_kernel.cu` CUDA 커널 작성
   - 공식: O = softmax(Q @ K^T / sqrt(d)) @ V
   - GQA 고려: KV 헤드 그룹화 (K/V heads = 2)
   
2. **PyTorch 바인딩**: `torch_binding.cpp`에 오버로드 추가
   ```cpp
   at::Tensor attention_forward(at::Tensor q, at::Tensor k, at::Tensor v) { ... }
   ```

3. **vLLM 통합**: `src/vllm_integration/executor_patch.py` LLaMA attention 후킹
   ```python
   # vllm/model_executor/layers/attention.py 의 _execute_attention 대체
   ```

4. **테스트**: `tests/test_attention_correctness.py`
   ```python
   # PyTorch baseline과 ±1e-5 오차 범위 검증
   ```

5. **2배 달성 검증**:
   ```bash
   python src/benchmark/benchmark_attention.py --target-speedup 2.0
   # 모든 배치/시퀀스에서 ≥2.0배 확인
   ```

### 성능 최적화 (RTX 1660 Super 기준)
1. **Baseline 측정**: PyTorch Flash Attention 성능 기록
2. **nvidia-nsys 프로파일링**:
   ```bash
   nvidia-nsys profile -o profile.nsys python src/benchmark/benchmark_attention.py
   # Memory bandwidth, compute utilization, kernel time 분석
   ```
3. **병목 지점 식별**:
   - Memory bandwidth < 75% → 메모리 최적화 우선
   - Compute < 30% → 알고리즘 개선
   - Kernel launch overhead > 10% → 배치 크기 증대

4. **CUDA 커널 튜닝**:
   ```cuda
   // 1) 블록 크기 스캔: 128, 256, 512
   // 2) Shared memory: 0KB (글로벌만), 4KB, 8KB, 16KB
   // 3) V-100/A-100에서 테스트 후 역최적화
   ```

5. **개선율 측정**: 각 변경 후 벤치마크 재실행하여 Δ speedup 계산

### vLLM 버전 업데이트 시 대응
1. 새 vLLM 버전 설치 후 호환성 검증
2. `src/vllm_integration/executor_patch.py`에서 deprecated API 확인
   - AttentionMetadata 구조 변경 여부
   - vLLM KVCache 레이아웃 변경 여부
3. `tests/test_integration.py::test_llama_3b_with_custom_attention` 실행
4. 필요시 커널 및 후킹 코드 수정

## ⚠️ 중요 주의사항: Custom Attention 개발

### CUDA 관련
- **CUDA 버전**: CUDA 12.1 권장 (vLLM 최신 호환)
- **Compute Capability**: RTX 1660 S = sm_75, RTX 2080 Ti = sm_75
  - 둘 다 동일 아키텍처이므로 커널 호환성 완벽 (배치/clock 다름)
- **메모리 안전성**: GPU 메모리 누수 필수 확인
  ```bash
  watch -n 0.5 nvidia-smi  # 지속 모니터링
  # 벤치마크 후 메모리 복귀 여부 확인
  ```

### 성능 측정
- **동기화**: 모든 벤치마크에서 `torch.cuda.synchronize()` 필수
  ```python
  # 잘못된 방법
  start = time.time(); kernel(); end = time.time()  # GPU와 비동기
  
  # 올바른 방법
  torch.cuda.synchronize(); start = time.perf_counter()
  kernel()
  torch.cuda.synchronize(); end = time.perf_counter()
  ```
- **Warmup 필수**: 처음 2-3회 실행은 제외 (캐시 워밍 & JIT)

### LLaMA 3.2 3B 특화
- **배치 크기**: RTX 1660 S는 배치 ≤8 권장 (메모리 6GB 제약)
- **시퀀스 길이**: 최대 8192 지원 (KVCache 축적에 주의)
- **Grouped Query Attention**: 커널에서 KV 헤드 그룹화 구현 필수
  ```
  num_heads=8, num_kv_heads=2 → 4개의 Q당 1개 KV
  ```

### 코드 품질
- **CUDA 커널 주석**: 메모리 레이아웃, 블록 구조, 복잡도 명시
  ```cuda
  // Memory layout: query [batch, num_heads, seq, head_dim] (batch-major)
  // Complexity: O(seq^2) for single batch, parallelized by batch & seq
  ```
- **에러 체킹**: cudaGetLastError() 또는 CHECK_CUDA 매크로 사용
- **테스트 커버리지**: 모든 배치/시퀀스 조합 최소 1회 실행

### RTX 1660 Super 제약사항
- **6GB VRAM 한계**: LLaMA 3.2 3B + KVCache로 ~8GB 필요
  - 해결책: gradient accumulation 없음, 배치 크기 제한
- **메모리 대역폭**: 334 GB/s (V100 대비 낮음, 최적화 필수)
- **레지스터 부족**: occupancy 50% 이상 유지 필수 (spillage 방지)
  ```cuda
  // 각 스레드 128 레지스터 이하 유지
  // nvidia-smi topo -m 으로 대역폭 확인
  ```
