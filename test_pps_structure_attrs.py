import paddleocr
from paddleocr import PPStructure

try:
    ppstructure = PPStructure(lang='en', layout=True, table=True, ocr=True, use_gpu=False, show_log=False)
    print("PPStructure 속성들:")
    print(dir(ppstructure))
    attrs = [attr for attr in dir(ppstructure) if 'struct' in attr.lower() or 'analy' in attr.lower()]
    print("구조/분석 관련 속성들:", attrs)
except Exception as e:
    print(f"오류 발생: {e}")
