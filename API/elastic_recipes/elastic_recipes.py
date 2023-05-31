from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, Q
import json

class ElasticRecipes:
    def __init__(self, password, cloud_id) -> None:
        self.client = Elasticsearch(
            cloud_id=cloud_id,
            http_auth=('elastic', password),
        )
    def create_index_if_not_exists(self, index_name):
        if not self.client.indices.exists(index_name):
            index_settings = {
                "settings": {
                    "analysis": {
                        "filter": {
                            "arabic_stemmer": {
                                "type": "stemmer",
                                "language": "arabic"
                            },
                            "arabic_stop": {
                                "type": "stop",
                                "stopwords": "_arabic_"
                            }
                        },
                        "analyzer": {
                            "rebuilt_arabic": {
                                "type": "custom",
                                "tokenizer": "standard",
                                "filter": ["lowercase", "decimal_digit", "arabic_stop", "arabic_normalization", "arabic_stemmer"]
                            }
                        }
                    }
                },
                "mappings": {
                    "properties": {
                        "Id": {
                            "type": "long"
                        },
                        "name": {
                            "type": "text",
                            "analyzer": "rebuilt_arabic",
                            "search_analyzer": "rebuilt_arabic"
                        },
                        "ingredients": {
                            "type": "text",
                            "analyzer": "rebuilt_arabic",
                            "search_analyzer": "rebuilt_arabic"
                        },
                        "steps": {
                            "type": "text",
                            "analyzer": "rebuilt_arabic",
                            "search_analyzer": "rebuilt_arabic"
                        },
                        "tags": {
                            "type": "text",
                            "analyzer": "rebuilt_arabic",
                            "search_analyzer": "rebuilt_arabic"
                        }
                    }
                }
            }
        # Create the index with the specified settings and mappings
        create_index_response = self.client.indices.create(index=index_name, body=index_settings)
        return create_index_response
          
    def get_recipe_by_id(self, index_name, id):
        search = Search(using=self.client, index=index_name).query("match", Id=id)
        response = search.execute()
        if response:
            return response[0].to_dict()
        else:
            return None
    
    def search_by_name(self, index_name, query):  
        search_query = Q(
            "multi_match",
            fields=["name^2", "tags", "steps", "ingredients"],
            query=query,
            fuzziness="AUTO"
        )
        search = self.client.search(index=index_name, body={"query": search_query.to_dict()})
        ids = []
        for hit in search['hits']['hits']:
            ids.append(hit['_source']['Id'])
        return ids

    def search_by_ingredients(self, index_name, query):
        search_query = Q(
            "multi_match",
            fields=["ingredients^2", "name", "steps"],
            query=query,
            fuzziness="AUTO"
        )
        search = self.client.search(index=index_name, body={"query": search_query.to_dict()})
        ids = []
        for hit in search['hits']['hits']:
            ids.append(hit['_source']['Id'])
        return ids

    def search_by_tags(self, index_name, query):
        search_query = Q(
            "multi_match",
            fields=["tags^2", "name"],
            query=query,
            fuzziness="AUTO"
        )
        search = self.client.search(index=index_name, body={"query": search_query.to_dict()})
        ids = []
        for hit in search['hits']['hits']:
            ids.append(hit['_source']['Id'])
        return ids
    
    def search_by_message(self, index, query):
        search_query = Q(
            "multi_match",
            fields=["name", "tags", "steps", "ingredients"],
            query=query,
            fuzziness="AUTO"
        )
        search = self.client.search(index=index, body={"query": search_query.to_dict()})
        ids = []
        for hit in search['hits']['hits']:
            ids.append(hit['_source']['Id'])
        return ids