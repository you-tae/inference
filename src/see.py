import numpy as np

# NumPy 배열 로드
logits = np.load("/home/doran/seld/src/inference/src/logits.npy")

# 출력 옵션 설정
np.set_printoptions(threshold=np.inf, linewidth=200)

# 문자열로 변환 후 저장
with open("logits_dump.txt", "w") as f:
    f.write(np.array2string(logits, threshold=np.inf, max_line_width=200))

print("✅ logits 전체 출력 내용을 logits_dump.txt에 저장했습니다.")
