# Counter-Strike 2 스킨 시장 매수 의사결정 모델

Counter-Strike 2 총기 스킨의 가격과 유동성을 분석하여 **오늘 실제로 매수할 만한 스킨이 있는지** 판단하는 머신러닝 연구 프로젝트입니다.

단순히 가격이 오를 스킨을 예측하지 않습니다. 모델 점수뿐 아니라 거래 비용, 유동성, 시장 상황, 가격 안정성, 체결 가능성을 함께 확인하고, 조건이 좋지 않으면 **거래하지 않음(No Trade)** 을 선택합니다.

> **현재 결론:** 연구 및 모의투자용으로만 사용할 수 있습니다. 실제 자금을 이용한 거래는 권장하지 않습니다.

## 프로젝트 핵심

이 프로젝트가 해결하려는 질문은 다음과 같습니다.

> 수수료와 위험을 감안하더라도 지금 매수할 가치가 있는 CS2 스킨이 있는가?

전체 분석 과정은 다음과 같습니다.

```text
SteamDT / CSFloat 데이터 수집
        ↓
가격·거래량·매물 데이터 정규화
        ↓
수익률·변동성·유동성·시장 지수 피처 생성
        ↓
거래 비용을 반영한 Triple Barrier 라벨링
        ↓
LightGBM 학습 및 워크포워드 검증
        ↓
수수료·체결 비용을 반영한 백테스트
        ↓
거절 정책 적용 → 매수 승인 또는 No Trade
        ↓
Streamlit 대시보드와 연구 보고서
```

## 주요 기능

- **시장 데이터 수집:** SteamDT 가격·거래량과 CSFloat 매물 스냅샷 수집
- **피처 생성:** 모멘텀, 변동성, 유동성, 데이터 최신성, 가격 급등락 비율 계산
- **시장 비교:** 동일 가중 CS2 시장 지수, 초과 수익률, 베타와 시장 상태 생성
- **라벨링:** 수수료와 슬리피지를 반영한 Triple Barrier 매수 결과 생성
- **모델 학습:** 기준 모델과 LightGBM 학습 및 시간 순서 기반 검증
- **현실적인 백테스트:** 거래 비용, 체결 실패, 투자 비중과 동시 보유 한도 반영
- **거절 정책:** 점수·유동성·가격·시장 조건이 부족한 후보 제외
- **결과 확인:** 매수 후보, 거절 이유, 성과와 데이터 상태를 대시보드로 제공

## 데이터와 분석 범위

### 데이터 출처

| 출처 | 사용 정보 |
| --- | --- |
| SteamDT | 스킨별 과거 가격, 거래량과 유동성 대용 지표 |
| CSFloat | 현재 매물 가격, 매물 수, 매물 깊이와 float 정보 |

### 현재 거래 대상

현재 연구 대상은 **총기 스킨**으로 제한합니다. 칼, 장갑, 케이스와 스티커 가격은 최종 거래 대상에 포함하지 않습니다.

CSFloat의 과거 시점별 스냅샷은 아직 충분하지 않습니다. 따라서 현재는 CSFloat를 핵심 매수 조건보다 유동성과 체결 가능성을 확인하는 보조 자료로 사용합니다.

## 모델의 판단 방식

각 스킨과 날짜는 가격, 거래량, 수익률, 변동성, 유동성, 시장 초과 수익률 등의 피처 벡터로 표현됩니다. LightGBM은 이 데이터를 이용해 거래 비용 이후에도 유효한 매수 기회일 가능성을 점수로 출력합니다.

높은 점수를 받은 후보도 다음 조건을 통과하지 못하면 거절됩니다.

- 유동성이 부족한 경우
- 가격 또는 데이터가 오래되거나 불안정한 경우
- 저가 스킨이라 거래 비용의 영향이 큰 경우
- 이벤트에 따른 일시적인 가격 변동인 경우
- 하락장에서 시장 대비 초과 성과가 없는 경우
- 투자 한도나 동시 보유 한도를 초과하는 경우

즉, 최종 출력은 단순한 스킨 순위가 아니라 **거래 승인 또는 거절 결정**입니다.

## 주요 연구 결과

거래 비용을 보정한 LightGBM 정책 비교 결과는 다음과 같습니다.

| 정책 | 승인 거래 수 | PnL | Profit Factor | 승률 |
| --- | ---: | ---: | ---: | ---: |
| 전체 가격 구간 | 704 | 0.6950 | 2.4118 | 66.76% |
| `$1+` | 416 | 0.1789 | 1.6882 | 71.15% |
| `$5+` | 195 | 0.0413 | 1.3241 | 77.44% |

전체 가격 구간의 성과는 좋아 보이지만 저가 스킨과 일부 기간에 의존할 수 있습니다. 더 엄격한 `$5+` 정책은 일부 지표를 개선했으나 기간별 안정성 기준을 통과하지 못했습니다.

따라서 현재 가장 타당한 결론은 다음과 같습니다.

- 파이프라인과 백테스트는 정상적으로 동작함
- 보수적인 `$5+` 정책은 추가 연구와 모의투자 후보로 사용할 수 있음
- 수익이 특정 아이템과 기간에 집중되어 있어 실거래 근거로는 부족함
- 장기간의 모의투자와 CSFloat 데이터 축적이 추가로 필요함

상세 결과는 [최종 제출 보고서](docs/final_submission_report.md)와 [최종 연구 보고서](docs/final_research_report.md)에서 확인할 수 있습니다.

## 빠른 시작

### 1. 실행 환경 만들기

Python 3.11 이상이 필요하며, 이 프로젝트는 Python 3.12 환경에서 검증했습니다.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.lock
.\.venv\Scripts\python.exe -m pip install -e .
```

### 2. API 키 설정하기

```powershell
Copy-Item .env.example .env
```

생성된 `.env` 파일에 필요한 키를 입력합니다.

```text
STEAMDT_API_KEY=...
CSFLOAT_API_KEY=...
```

`.env`와 개인 API 키는 Git에 커밋하지 마십시오.

### 3. 파이프라인 확인하기

실행할 단계를 먼저 확인하려면 다음 명령을 사용합니다.

```powershell
.\.venv\Scripts\python.exe -m cs_market_model.pipeline.run_all --dry-run
```

기존 로컬 데이터를 이용해 전체 분석을 실행합니다.

```powershell
.\.venv\Scripts\python.exe -m cs_market_model.pipeline.run_all
```

API 데이터 수집부터 포함하려면 다음과 같이 실행합니다.

```powershell
.\.venv\Scripts\python.exe -m cs_market_model.pipeline.run_all --include-collectors
```

생성되는 `data/`와 `reports/`의 주요 결과 파일은 의도적으로 Git에서 제외됩니다.

## 대시보드 실행

```powershell
.\.venv\Scripts\streamlit.exe run src/cs_market_model/dashboard/app.py
```

대시보드에서는 다음 내용을 확인할 수 있습니다.

- 현재 매수 승인 후보
- 거절된 후보와 거절 사유
- 시장 상승·중립·하락 상태
- 포트폴리오 성과와 낙폭
- 거래 용량 제한 검증 결과
- 모델 설명과 데이터 최신 상태

## 주요 연구 명령어

```powershell
# 저가 스킨 정책 비교
.\.venv\Scripts\python.exe -m cs_market_model.research.day22_low_price_policy

# 수익의 아이템·기간 집중도 분석
.\.venv\Scripts\python.exe -m cs_market_model.research.day23_policy_attribution

# $5+ 정책의 거래 용량 검증
.\.venv\Scripts\python.exe -m cs_market_model.research.day24_capacity_sensitivity

# 강화된 모의투자 거절 정책 검증
.\.venv\Scripts\python.exe -m cs_market_model.research.day25_paper_trade_rejection

# CSFloat 데이터 커버리지 확인
.\.venv\Scripts\python.exe -m cs_market_model.research.day19_csfloat_coverage
```

## 테스트와 품질 검사

```powershell
# 전체 테스트
.\.venv\Scripts\python.exe -m pytest

# 코드 스타일 및 정적 검사
.\.venv\Scripts\python.exe -m ruff check .

# API 키와 비밀 정보 검사
.\.venv\Scripts\python.exe -m cs_market_model.security.scan_secrets
```

## 프로젝트 구조

```text
Counter_Strike_Market_Model/
├─ configs/                 # 데이터 범위와 백테스트 정책 설정
├─ docs/                    # 연구·제출·재현 문서
├─ notebooks/               # 진단과 시각화 노트북
├─ reports/                 # 로컬 분석 결과와 백테스트 보고서
├─ src/cs_market_model/
│  ├─ collectors/           # SteamDT·CSFloat 수집기
│  ├─ normalization/        # 가격·아이템·매물 데이터 정규화
│  ├─ features/             # 피처와 CS2 시장 지수 생성
│  ├─ labeling/             # Triple Barrier 라벨링
│  ├─ models/               # 기준 모델과 LightGBM
│  ├─ backtesting/          # 포트폴리오·거래 비용·거절 정책
│  ├─ research/             # 강건성 및 모의투자 분석
│  ├─ dashboard/            # Streamlit 대시보드
│  └─ pipeline/             # 전체 실행 파이프라인
└─ tests/                   # 단위 및 통합 테스트
```

## 현재 한계

- CSFloat의 장기 point-in-time 데이터가 충분하지 않음
- 일부 수익이 특정 아이템과 기간에 집중됨
- 실제 주문에 따른 가격 충격과 체결 수량을 완전히 재현하지 못함
- 총기 스킨만 거래 대상으로 사용함
- 자동 실거래와 실제 자금 승인 기능을 제공하지 않음

재현 방법과 생성 파일은 [REPRODUCE.md](docs/REPRODUCE.md)를 참고하십시오.

## 주의 사항

이 저장소는 교육 및 연구 목적으로 제작되었습니다. 결과는 투자 조언이 아니며, 과거 백테스트 성과는 미래 수익을 보장하지 않습니다. 충분한 장기 모의투자와 추가 검증 전에는 실제 자금을 사용하지 마십시오.
