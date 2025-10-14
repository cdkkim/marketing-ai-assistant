import os
import json
import time
import csv
import google.generativeai as genai

def call_gemini(prompt: str, model_name="gemini-1.5-pro", temperature=0.6, max_tokens=2048):
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("환경변수 GOOGLE_API_KEY 가 설정되어 있지 않습니다.")
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(model_name)
    generation_config = {"temperature": temperature, "max_output_tokens": max_tokens}
    resp = model.generate_content(prompt, generation_config=generation_config)
    return resp.text if hasattr(resp, "text") else str(resp)

def call_with_retry(prompt, retries=3):
    for attempt in range(retries):
        try:
            return call_gemini(prompt)
        except Exception as e:
            print(f"[WARN] API 호출 실패 {attempt+1}/{retries}: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise

if __name__ == "__main__":
    # 1. persona_generator에서 저장한 JSON 불러오기
    with open("personas.json", "r", encoding="utf-8") as f:
        personas = json.load(f)

    # 2. 결과 CSV 저장
    with open("gemini_responses.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["업종","프랜차이즈여부","점포연령","고객연령대","고객행동","prompt","gemini_response"])

        for idx, p in enumerate(personas, 1):
            print(f"[{idx}/{len(personas)}] {p['업종']} / {p['프랜차이즈여부']} / {p['점포연령']} / {p['고객연령대']} / {p['고객행동']}")
            response = call_with_retry(p["prompt"])
            writer.writerow([p["업종"], p["프랜차이즈여부"], p["점포연령"], p["고객연령대"], p["고객행동"], p["prompt"], response])

    print("[DONE] Gemini 응답이 gemini_responses.csv 에 저장되었습니다.")
