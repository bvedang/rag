from rerankers import Reranker
from rerankers.documents import Document

ranker = Reranker("answerdotai/answerai-colbert-small-v1", model_type='colbert')

docs = [
    Document(text='Hayao Miyazaki is a Japanese director, born on January 5, 1941. He is best known for his animated feature films, including Spirited Away.'),
    Document(text='Walt Disney is an American author, director and entrepreneur, known for creating Mickey Mouse and founding The Walt Disney Company.'),
    Document(text='Spirited Away is a 2001 Japanese animated fantasy film written and directed by Hayao Miyazaki.')
]

query = 'Who directed spirited away?'

if ranker:
    results = ranker.rank(query=query, docs=docs)


for result in results:
        print(f"Score: {result.score:.4f}")
        print(f"Text: {result.text}")
        print("-" * 50)
