import yaml
from openai import OpenAI
from neo4j import GraphDatabase


class RAGNeo4j:
    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        self.neo4j_driver = GraphDatabase.driver(
            self.config['target']['neo4j_uri'],
            auth=(self.config['target']['username'], self.config['target']['password'])
        )


    def generate_cypher_query(self, question):
        prompt = f"""
        Given the following graph structure:
        {self.config['target']['graph_structure']}

        And the user's question: "{question}"

        Generate a Cypher query to retrieve relevant information from the graph and ONLY return the cypher query in text format. DO NOT ADD ```cypher
        """

        client = OpenAI(api_key=self.config['embedding']['api_key'])
        response = client.chat.completions.create(
            model='gpt-4o',
            messages=[
                {"role": "system",
                 "content": "You are a helpful assistant that generates Cypher queries based on natural language questions and a given graph structure."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8
        )

        return response.choices[0].message.content

    def execute_cypher_query(self, query):
        with self.neo4j_driver.session() as session:
            print("### EXECUTING CYPHER QUERY ###")
            print(query)
            result = session.run(query)
            return [record.data() for record in result]

    def generate_answer(self, question, query_results):
        prompt = f"""
        Question: {question}

        Graph database results: {query_results}

        Please provide a concise answer to the question based on the given information.
        """

        system_message = "You are a helpful assistant that answers questions about fashion clothing based on the provided graph database results."
        client = OpenAI(api_key=self.config['embedding']['api_key'])
        response = client.chat.completions.create(
            model='gpt-4o',
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8
        )

        return response.choices[0].message.content

    def answer_question(self, question):
        cypher_query = self.generate_cypher_query(question)
        query_results = self.execute_cypher_query(cypher_query)
        answer = self.generate_answer(question, query_results)
        return answer

    def close(self):
        self.neo4j_driver.close()


# Example usage
if __name__ == "__main__":
    config_path = "file_to_neo4j.yaml"
    rag = RAGNeo4j(config_path)

    while True:
        question = input("Enter your question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break

        try:
            answer = rag.answer_question(question)
            print(f"\nQuestion: {question}")
            print(f"Answer: {answer}\n")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

    rag.close()
    print("Thank you for using the RAG system. Goodbye!")

