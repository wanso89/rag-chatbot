from elasticsearch import Elasticsearch


es = Elasticsearch("http://172.10.2.70:9200")
INDEX_NAME = "rag_documents_kure_v1"


prefix = "NDT User Guide(Win).pdf_"

# 1. 해당 prefix로 시작하는 문서들을 _id 기준으로 수집
res = es.search(index=INDEX_NAME, body={
    "query": {
        "match_all": {}
    },
    "_source": False,
    "size": 1000  # 필요 시 더 늘리기
})

ids_to_delete = [doc["_id"] for doc in res["hits"]["hits"] if doc["_id"].startswith(prefix)]

print(f"🧹 삭제 대상 문서 수: {len(ids_to_delete)}")

# 2. 일괄 삭제
#for doc_id in ids_to_delete:
#    es.delete(index=INDEX_NAME, id=doc_id)
#    print(f"🗑️  Deleted: {doc_id}")
#
