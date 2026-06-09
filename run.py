import csv
import ir_datasets
from rag_agent import RAGAgent

csv_filename = "answers.csv"

if __name__ == '__main__':
    dataset = ir_datasets.load("cord19/fulltext/trec-covid")
    agent = RAGAgent(dataset)

    with open(csv_filename, mode='w', encoding='utf-8-sig', newline='') as csv_file:
        # Força aspas em todos os campos para que vírgulas ou pipes internos não quebrem as colunas
        writer = csv.writer(csv_file, delimiter=';', quoting=csv.QUOTE_ALL)

        writer.writerow(["question", "answer"])

        with open('questions.txt', 'r', encoding='utf-8') as f:
            for line in f:
                question_raw = line.strip()

                if question_raw:
                    # Extrai apenas o texto da pergunta (removendo o "ID: X | Pergunta:")
                    if "Pergunta: " in question_raw:
                        question = question_raw.split("Pergunta: ")[-1]
                    else:
                        question = question_raw

                    print(f"\n=========================================")
                    print(f"Processing: {question}")

                    answer = agent.ask(question)

                    clean_answer = answer.replace('\n', ' ')
                    print(clean_answer)

                    writer.writerow([question_raw, clean_answer])
                    csv_file.flush()
                    
exit(0)