# vLLM 커스텀 연산 최적화 프로젝트

## 📋 프로젝트 개요

vLLM(Large Language Model serving engine) 최신 버전에 간단한 커스텀 연산을 추가하고, 이를 성능 최적화하는 연구 프로젝트입니다. 목표는 vLLM의 확장성과 최적화 기법을 학습하면서 실제 production-level의 성능 개선을 달성하는 것입니다.

## 🎯 프로젝트 목표

1. **vLLM 아키텍처 이해**: 코어 컴포넌트, 데이터 플로우, 최적화 기법 학습
2. **커스텀 연산 추가**: vLLM에 새로운 연산 커널 또는 기능 구현
3. **성능 최적화**: 커스텀 연산의 성능 벤치마킹 및 개선
4. **프로덕션 레벨 품질**: 테스트, 문서화, CI/CD 통합

## 📑 단계별 개발 계획

### Phase 1: 환경 구성 및 기초 학습 (1-2주)
- [ ] vLLM 최신 버전 설치 및 빌드
- [ ] vLLM 코드베이스 구조 분석
  - `vllm/` - 메인 코드
  - `vllm/model_executor/` - 모델 실행 엔진
  - `vllm/worker/` - GPU 워커
  - `csrc/` - CUDA 커널 구현
- [ ] 기본 추론(inference) 파이프라인 이해
- [ ] 벤치마킹 도구 설정 (데이터 수집, 메트릭 정의)

### Phase 2: 커스텀 Attention 연산 설계 (1주)
- [ ] **Custom Attention 알고리즘 설계**
  - LLaMA 3.2의 Grouped Query Attention (GQA) 기반 최적화
  - 목표: flash-attention 레벨의 메모리 효율성
  - 로컬 윈도우 어텐션 또는 sparse attention 검토
- [ ] PyTorch 구현 (검증용 baseline)
  ```python
  # 목표 인터페이스
  output = custom_attention(
      query, key, value,      # [batch, seq_len, hidden_dim]
      mask=None,
      dropout=0.0,
      scale=None
  )
  ```
- [ ] 성능 기준 설정: **PyTorch baseline 기준 2배 속도 달성**

### Phase 3: Custom Attention 커널 구현 (2-3주)
- [ ] **CUDA 커널 작성** (`src/custom_ops/custom_attention_kernel.cu`)
  - GQA (Grouped Query Attention) 구현
  - 메모리 coalescing 최적화
  - Shared memory 활용 (vs global memory)
  - 타겟 블록 크기: 32×8 또는 64×4 (RTX 1660 Super 기준)
- [ ] **vLLM LLaMA 3.2 통합**
  - `vllm/model_executor/layers/attention.py` 후킹
  - vLLM의 KVCache 구조 호환성
  - Batch processing 지원
- [ ] **단위 테스트** (`tests/test_attention_correctness.py`)
  - LLaMA 3.2 config 기준 (hidden_size=3072, num_heads=8, num_kv_heads=2)
  - 다양한 seq_length 테스트: [128, 512, 2048, 8192]
- [ ] 정확성 검증: PyTorch baseline과 **±1e-5 오차 범위**

### Phase 4: 성능 최적화 - 2배 달성 목표 (2-3주)
- [ ] **프로파일링 (nvidia-nsys 사용)**
  - RTX 1660 Super 메모리 대역폭: ~336 GB/s
  - 목표 메모리 활용률: >80%
  - RTX 2080 Ti 비교 벤치: >90% 스케일링
- [ ] **CUDA 커널 튜닝**
  - 블록 크기 스캔: 64, 128, 256, 512 threads
  - Shared memory 사용량 감소 (쿼리 블로킹)
  - 레지스터 압박 회피 (occupancy >50% 유지)
  - 분기 최소화 (divergence 감소)
- [ ] **배치 크기 적응형 최적화**
  - 배치 크기별 최적 커널 설정 저장
  - vLLM scheduler와 협력하여 배치 크기 조절
- [ ] **2배 속도 달성 검증 체크리스트**
  - [ ] Latency: <0.5ms/token (1660 S) 또는 <0.3ms/token (2080 Ti)
  - [ ] Throughput: >2000 tokens/sec
  - [ ] Memory: <4GB 사용 (1660 S 기준)

### Phase 5: 검증 및 Phase 2 계획 (1-2주)
- [ ] **LLaMA 3.2 3B 종합 벤치마크**
  - Seq lengths: [128, 512, 2048, 8192]
  - Batch sizes: [1, 4, 8, 16, 32]
  - 두 GPU 환경 모두 측정
- [ ] **2배 성능 달성 확인**
  - PyTorch baseline vs Custom CUDA 비율 계산
  - 모든 배치/시퀀스 조합에서 ≥2배 확인
- [ ] 성능 비교 분석 및 리포트 작성 (`docs/BENCHMARK_RESULTS.md`)
- [ ] 코드 리뷰 및 정제
- [ ] **Phase 2 준비: Quantization 최적화**
  - Custom Attention 통합 후 INT8/FP8 량화 추가

## 🛠️ 개발 환경 설정

### 필수 요구사항
```bash
# 시스템 요구사항
# - GPU: RTX 1660 Super (6GB VRAM) 또는 RTX 2080 Ti (11GB VRAM)
# - Python 3.10+
# - CUDA 12.1 (vLLM 최신 버전 호환)
# - cuDNN 8.9+

# 1. vLLM 최신 버전 설치
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout main  # 최신 버전 사용

# 2. 개발 환경 설정
pip install -e .  # 개발 모드 설치
pip install -e ".[dev]"  # 테스트/개발 도구 포함
pip install ninja  # CUDA 컴파일 가속

# 3. LLaMA 3.2 3B 모델 다운로드 (HuggingFace 사용)
python -m pip install huggingface-hub
huggingface-cli download meta-llama/Llama-2-3b-hf --local-dir ./models/llama-3.2-3b

# 4. 프로젝트 서브모듈 설정
git submodule update --init --recursive
```

### GPU 메모리 사전 점검
```bash
# 현재 GPU 상태 확인
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# LLaMA 3.2 3B 메모리 요구사항:
# - 모델 가중치: ~6GB (FP16)
# - KVCache (배치 8, seq_len 8192): ~1.5GB
# - 작업 메모리: ~0.5GB
# 총 권장: ~8GB (1660 Super는 6GB로 제한)
```

### 빌드 및 테스트
```bash
# CUDA 커널 컴파일 (Custom Attention)
cd src/custom_ops
ninja build  # 또는 python setup.py build_ext --inplace
cd ../..

# 정확성 검증 (LLaMA 3.2 3B 기준)
pytest tests/test_attention_correctness.py -v

# 벤치마크: baseline vs custom attention
python src/benchmark/benchmark_custom_op.py --batch-size 8 --seq-length 2048

# 성능 프로파일링 (RTX 1660 Super 기준)
nvidia-nsys profile -o ./results/attention_profile python src/benchmark/benchmark_custom_op.py

# 통합 테스트 (vLLM 내 custom attention 검증)
pytest tests/test_integration.py::test_llama_3b_with_custom_attention
```

## 📚 핵심 개념 및 가이드라인

### vLLM 아키텍처 핵심
- **Scheduler**: 요청 배치 및 프리엠션 관리
- **Model Executor**: GPU에서 모델 실행
- **Worker**: GPU별 프로세스, 병렬 처리 조율
- **OptimizedAttention**: FlashAttention 등 최적화된 주의 메커니즘

### 코딩 컨벤션
- CUDA 커널: `vllm/csrc/` 에 `.cu`, `.cuh` 파일로 구현
- Python binding: pybind11 사용 (`vllm/csrc/torch_bindings.cpp`)
- 테스트: `tests/` 디렉토리, `test_*.py` 네이밍
- 성능 측정: latency, throughput, memory usage 3가지 지표 수집

### 최적화 체크리스트
- [ ] 메모리 접근 패턴 (coalesced memory access)
- [ ] 스레드 활용 (occupancy 계산)
- [ ] 분기 제거 (branch divergence 최소화)
- [ ] 동기화 오버헤드 감소
- [ ] 배치 크기 적응형 튜닝

## 📁 프로젝트 구조

```
personal-multi-agent/
├── README.md                          # 이 파일
├── .github/
│   └── copilot-instructions.md        # AI 개발 가이드
├── src/
│   ├── custom_ops/                    # 커스텀 연산 구현
│   │   ├── __init__.py
│   │   ├── custom_kernel.cu           # CUDA 커널
│   │   ├── custom_kernel.cuh          # CUDA 헤더
│   │   ├── torch_binding.cpp          # PyTorch 바인딩
│   │   └── python_interface.py        # Python 인터페이스
│   ├── vllm_integration/              # vLLM 통합 코드
│   │   ├── executor_patch.py          # Executor 확장
│   │   └── scheduler_patch.py         # Scheduler 통합
│   └── benchmark/                     # 벤치마킹 도구
│       ├── benchmark_custom_op.py
│       ├── profile_kernel.py
│       └── compare_baselines.py
├── tests/
│   ├── test_correctness.py            # 정확성 테스트
│   ├── test_performance.py            # 성능 테스트
│   └── test_integration.py            # 통합 테스트
├── docs/
│   ├── ARCHITECTURE.md                # 설계 문서
│   ├── CUDA_KERNEL_GUIDE.md           # CUDA 커널 개발 가이드
│   ├── BENCHMARK_RESULTS.md           # 벤치마크 결과
│   └── IMPLEMENTATION_LOG.md           # 구현 로그
└── scripts/
    ├── setup_dev_env.sh               # 개발 환경 설정
    ├── build_cuda.sh                  # CUDA 빌드
    └── run_benchmarks.sh              # 벤치마크 실행
```

## 🔍 핵심 파일 및 통합 포인트

### vLLM 내 주요 파일
- `vllm/executor/executor.py` - 실행 엔진 기본 인터페이스
- `vllm/model_executor/gpu_executor.py` - GPU 실행 구현
- `vllm/worker/gpu_worker.py` - 워커 구현
- `vllm/attention/` - 어텐션 최적화 구현

### 커스텀 연산 통합
1. CUDA 커널 구현 → `src/custom_ops/`
2. PyTorch 바인딩 작성 → `torch_binding.cpp`
3. Python 래퍼 생성 → `python_interface.py`
4. vLLM Executor에 후킹 → `vllm_integration/`

## ⚙️ 성능 측정 방법론

### 지표
- **Latency (ms)**: 단일 요청 처리 시간
- **Throughput (tokens/sec)**: 초당 처리 토큰 수
- **Memory (GB)**: 피크 GPU 메모리 사용량
- **Speedup**: baseline 대비 개선율

### 벤치마킹 자동화
```python
# benchmark/benchmark_custom_op.py 예상 구조
def benchmark(batch_size, seq_length, num_trials=100):
    warmup()
    times = []
    for _ in range(num_trials):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = custom_op(input_tensor)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return compute_statistics(times)
```

## 📊 성능 목표 및 예상 결과

### Phase 1 목표 달성 기준 (Custom Attention)

| 지표 | 대상 | RTX 1660 Super | RTX 2080 Ti |
|------|-----|---|---|
| **Speedup** | PyTorch baseline 대비 | **≥2.0x** | ≥2.0x |
| **Latency** | 단일 토큰 (배치1, seq 2048) | <0.5ms | <0.3ms |
| **Throughput** | 초당 토큰 처리 | >2000 tok/s | >4000 tok/s |
| **Memory** | 피크 게이지 (배치8) | <4.5GB | <8GB |
| **정확성** | PyTorch와의 오차 | ±1e-5 | ±1e-5 |

### Phase 2 목표 (Quantization 최적화)
- INT8 양화 후 추가 1.5배 속도 개선
- 정확성 손상 <1% (BLEU score 기준)

### 최종 목표
- [ ] Custom Attention: **2배 달성**
- [ ] Custom Attention + INT8: **3배 달성**
- [ ] 프로덕션 배포 가능한 코드 품질

## 📝 참고 자료

### LLaMA 3.2 3B 모델 설정
- [Meta LLaMA 3.2 GitHub](https://github.com/meta-llama/llama)
- [HuggingFace LLaMA 3.2 3B](https://huggingface.co/meta-llama/Llama-2-3b)
- **주요 파라미터**: hidden_size=3072, num_heads=8, num_kv_heads=2 (GQA)

### 최적화 참고 자료
- [vLLM 공식 문서](https://docs.vllm.ai/) - Executor, Scheduler 구조
- [Flash Attention](https://github.com/Dao-AILab/flash-attention) - Attention 최적화 기준
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [nvidia-nsys 프로파일링](https://docs.nvidia.com/nsys/)

### GPU 환경
- **RTX 1660 Super**: 1408 CUDA cores, 6GB VRAM, sm_75 compute capability
- **RTX 2080 Ti**: 4352 CUDA cores, 11GB VRAM, sm_75 compute capability
- [nvidia-nsys 메모리 분석](https://docs.nvidia.com/nsys/profiling-guide/index.html#memory-analysis)

### 대안 접근법
- [Triton GPU Programming](https://openai.com/triton/) - CUDA 대체 고려
- [CuPy](https://cupy.dev/) - NumPy 스타일 GPU 프로그래밍
