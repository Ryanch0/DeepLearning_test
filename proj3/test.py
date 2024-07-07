
# STEP 1 import module
from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel

#STEP 2 create task model
summarizer = pipeline("summarization", model="psyche/KoT5-summarization") #요약모델

model_name = "skt/kogpt2-base-v2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
device = 0 if torch.cuda.is_available() else -1
generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=device)


from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

def generate_clickbait_title(summary):
    prompt = f"요약된 뉴스 내용: {summary}\n자극적이고 과장된 제목: "
    generated = generator(prompt, max_length=500, num_return_sequences=1, no_repeat_ngram_size=2)[0]['generated_text']
    
    # 결과에서 제목 부분만 추출
    title = generated.split("자극적이고 과장된 제목: ")[-1].strip()
    return title


class Input_token(BaseModel):
    body : str

@app.post("/sum")
async def summary(input_token : Input_token):
    input_dict = input_token.dict()
    #STEP 3 load input_token value
    # text = "조 바이든(81) 미국 대통령이 대선 후보 사퇴론을 잠재우기 위해 주말을 전후해 연이은 언론 인터뷰, 공개 유세 등에 나섰지만 건강 우려를 불식시키기엔 역부족이었다는 평가가 나온다. 현실과 다소 동떨어진 인식을 드러내는가 하면 인터뷰 질문 사전 조율 논란에 휩싸이는 등 오히려 논란을 키웠다는 지적도 있다.\
    # 민주당 내 확산되고 있는 공개적 사퇴론, ‘비(非)바이든’으로 돌아선 진보 성향의 주류 언론, 등 돌리고 있는 고액 기부자 등 바이든이 처한 상황은 그야말로 사면초가다. 그럼에도 바이든 대통령이 연일 ‘완주’ 의지를 보이는 데다 11월 대선까지 물리적 시간이 충분치 않은 상황에서 갑작스런 ‘환승’이 공멸을 부를 거란 반론도 만만치 않다. 사퇴론과 완주론이 첨예하게 맞선 형국을 두고 미 월스트리트저널(WSJ)은 “양쪽 간 물러설 수 없는 '치킨게임'이 벌어지고 있다”고 짚었다."
    #STEP 4 summarize
    result = summarizer(input_token.body)
    summary_text = result[0]["summary_text"]

    trash_title = generate_clickbait_title(summary_text)

    #STEP 5 process the result
    # print(result)
    input_dict.update({"summary" : summary_text})
    input_dict.update({"trash_title" : trash_title})
    return input_dict









