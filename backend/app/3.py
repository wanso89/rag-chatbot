from elasticsearch import Elasticsearch

es = Elasticsearch("http://172.10.2.70:9200")
INDEX_NAME = "rag_documents_kure_v1"


# 전체 문서에서 _id가 특정 문자열을 포함하는 것만 필터링
res = es.search(index=INDEX_NAME, body={
    "query": {
        "match_all": {}
    },
    "_source": False,
    "size": 10000  # 필요 시 조절
})

#prefix = "7b7b14a5-df9b-4295-8213-b01a75151fcd_한국도로공사서비스 2025년 경력직 전문직 제한경쟁 공개채용 공고_pdf_"
#prefix = "eef7daa8-3397-4b69-a681-343d3da1925b_Cloudera 운영자 메뉴얼_pdf_"
prefix = "Cloudera 운영자 메뉴얼_pdf_"
ids = [doc["_id"] for doc in res["hits"]["hits"] if prefix in doc["_id"]]

print(f"🔎 검색된 문서 수: {len(ids)}")
for _id in ids:
    print(f"🧾 {_id}")

