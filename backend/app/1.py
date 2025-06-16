from elasticsearch import Elasticsearch

es = Elasticsearch("http://172.10.2.70:9200")
INDEX_NAME = "rag_documents_kure_v1"


file_name = "NDT User Guide(Win).pdf"  # 실제 삭제 시 사용하는 값

# 실제 Elasticsearch 내부 document_id 비교
res = es.search(index=INDEX_NAME, body={
    "query": {
        "match_all": {}
    },
    "_source": ["document_id"],
    "size": 100
})

print(f"🔍 삭제 시 사용한 file_name: '{file_name}'")
print("📦 실제 ES 저장된 document_id 값들:")
for hit in res["hits"]["hits"]:
    doc_id = hit["_id"]
    doc_docid = hit["_source"].get("document_id")
    print(f" - _id: {doc_id}, document_id: '{doc_docid}' --> 일치: {doc_docid == file_name}")

