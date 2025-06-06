import os
from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain # Cadeia específica para Q&A com Cypher
from langchain.prompts import PromptTemplate

API_KEY_groq = ""
API_KEY_openai = ""
# --- 1. Configuração ---
# Defina suas variáveis de ambiente (ou configure diretamente)
# os.environ["OPENAI_API_KEY"] = "API_KEY_openai" # Ou sua chave da OpenAI
# os.environ["NEO4J_URI"] = "bolt://localhost:7687" # Ou seu URI do AuraDB/Neo4j
# os.environ["NEO4J_USERNAME"] = "neo4j"
# os.environ["NEO4J_PASSWORD"] = "SUA_SENHA_NEO4J"

# Verifique se as variáveis de ambiente estão configuradas
# if not os.getenv("OPENAI_API_KEY") or not os.getenv("NEO4J_URI"):
#     print("Por favor, defina as variáveis de ambiente OPENAI_API_KEY, NEO4J_URI, NEO4J_USERNAME, e NEO4J_PASSWORD.")
#     exit()

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0, api_key = API_KEY_openai) # Use um modelo poderoso para tradução para Cypher

# --- 2. Conectar ao Neo4j e obter esquema (LangChain faz isso internamente para GraphCypherQAChain) ---
try:
    graph = Neo4jGraph(
        url="bolt://localhost:7687",
        username="neo4j",
        password="password"
    )
    # Você pode inspecionar o esquema se desejar:
    # print("Schema do Grafo:")
    # print(graph.schema)
    print("Conectado ao Neo4j com sucesso!")

    # Exemplo de dados para popular (execute uma vez se o banco estiver vazio)
    # graph.query("""
    # MERGE (p:Person {name: "Alice", age: 30})
    # MERGE (c:Company {name: "Acme Corp"})
    # MERGE (city:City {name: "Wonderland"})
    # MERGE (p)-[:WORKS_AT {role: "Engineer"}]->(c)
    # MERGE (c)-[:LOCATED_IN]->(city)

    # MERGE (b:Person {name: "Bob", age: 25})
    # MERGE (startup:Company {name: "StartupX"})
    # MERGE (b)-[:WORKS_AT {role: "Designer"}]->(startup)
    # MERGE (startup)-[:LOCATED_IN]->(city)

    # MERGE (m:Movie {title: "The Matrix", released: 1999})
    # MERGE (k:Person {name: "Keanu Reeves"})
    # MERGE (k)-[:ACTED_IN {role: "Neo"}]->(m)
    # """)
    # print("Dados de exemplo carregados/verificados.")

except Exception as e:
    print(f"Erro ao conectar ou popular o Neo4j: {e}")
    graph = None
    exit()


# --- 3. Construir a Cadeia de NL para Cypher (GraphCypherQAChain simplifica isso) ---
# GraphCypherQAChain combina a geração de Cypher e a Q&A em uma só cadeia.
# Ela usa o schema do grafo para informar a geração da query Cypher.

# Você pode customizar os prompts se necessário.
# Este é o prompt padrão para geração de Cypher (simplificado para ilustração)
CYPHER_GENERATION_TEMPLATE = """
Você é um expert em Neo4j e Cypher. Dada uma pergunta em linguagem natural e um esquema de grafo,
gere uma query Cypher para responder à pergunta. NÃO RESPONDA A PERGUNTA, apenas gere a query.
Não use exemplos na query, apenas o que for necessário com base na pergunta e no esquema.
Não coloque a query dentro de blocos de código markdown. Retorne apenas a query Cypher.

Esquema do Grafo:
{schema}

Pergunta: {question}
Query Cypher:
"""
CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
)

# Este é o prompt para gerar a resposta final com base nos resultados da query
QA_TEMPLATE = """Você é um assistente que responde perguntas com base em um contexto fornecido.
O contexto é o resultado de uma query Cypher em um grafo de conhecimento.
Use a informação do contexto para responder à pergunta. Seja conciso e direto.
Se a informação não estiver no contexto, diga que não encontrou a resposta no grafo.

Contexto:
{context}

Pergunta: {question}
Resposta útil:"""
QA_PROMPT = PromptTemplate(input_variables=["context", "question"], template=QA_TEMPLATE)

# --- 4, 5, 6: Executar Query, Formatar, Gerar Resposta (tudo encapsulado pela cadeia) ---
if graph:
    try:
        # `validate_cypher=True` (padrão) pode tentar corrigir pequenas falhas na query Cypher gerada usando o LLM.
        # `return_intermediate_steps=True` permite ver a query Cypher gerada.
        chain = GraphCypherQAChain.from_llm(
            graph=graph,
            llm=llm,
            cypher_prompt=CYPHER_GENERATION_PROMPT, # Opcional, para customizar
            qa_prompt=QA_PROMPT,                   # Opcional, para customizar
            validate_cypher=True,
            verbose=True, # Mostra os passos intermediários, incluindo a query Cypher
            return_intermediate_steps=True,
            allow_dangerous_requests=True
            # top_k=5 # Limita o número de resultados do grafo a serem passados como contexto
        )
        print("Cadeia GraphCypherQAChain criada.")
    except Exception as e:
        print(f"Erro ao criar a GraphCypherQAChain: {e}")
        chain = None
        exit()


    # --- 7. Testar a Cadeia ---
    questions = [
        "Quantas pessoas fizeram parte do filme Matrix?",
        "Com quantas pessoas Keanu Reeves atuou em seus filmes?",
        "Qual a idade de Demi?",
        "Em quais filmes Keanu Reeves já atuou?",
        "Qual a idade da Keanu Reeves?",
        "Existe algum filme chamado 'Innovate Ltd'?" # Pergunta que não deve encontrar dados
    ]

    if chain:
        for question_text in questions:
            print(f"\n\n Perguntando: {question_text}")
            try:
                # Para LangChain > 0.1.0, use `chain.invoke`
                result = chain.invoke({"query": question_text})

                print("\n> Query Cypher Gerada (passo intermediário):")
                if result.get("intermediate_steps") and len(result["intermediate_steps"]) > 0:
                    print(result["intermediate_steps"][0].get("query", "Não disponível")) # A query está no primeiro passo
                else:
                    print("Query intermediária não encontrada.")

                print("\n> Contexto do Grafo (passo intermediário):")
                if result.get("intermediate_steps") and len(result["intermediate_steps"]) > 0:
                     print(result["intermediate_steps"][0].get("context", "Não disponível")) # O contexto está no primeiro passo

                print(f"\n> Resposta Final: {result['result']}")

            except Exception as e:
                print(f"Erro ao processar a pergunta '{question_text}': {e}")
                import traceback
                traceback.print_exc()
    else:
        print("A cadeia não foi inicializada devido a erros anteriores.")

else:
    print("A conexão com o Neo4j não foi estabelecida. Encerrando.")