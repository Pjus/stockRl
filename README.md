# stockRl
stock reinforcement, flask web service
강화학습을 이용한 주식 매매 행동 어드바이저
## 전체 과정
1. 주식 데이터 수집 yahoo finance (pandas_datareader)
2. 인디케이터 계산
3. 스케일링
4. 강화학습 환경 구성
5. openAI baseline 알고리즘 사용
6. 강화학습 실행
7. flask로 구동
8. goorm ide로 웹 서비스 
9. javascript chart 생성
## 기술
* 참조 사이트 : 
1. https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e
2. https://mclearninglab.tistory.com/64?category=869053
### 강화학습 과정
1. gym custom environment 구성
2. openAI baseline PPO2, A2C 알고리즘 사용
3. MLP, LSTM 딥러닝 모델 사용
4. 
## 결과
![profit](/./results/profit.png)
![profit](/./results/test.png)
