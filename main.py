from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
# from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama


load_dotenv()


def main():
    print("Hello from langchain-course!")

    information = """
Naruto[a] is a Japanese manga series written and illustrated by Masashi Kishimoto. It tells the story of Naruto Uzumaki, a young, socially isolated ninja who seeks recognition from his peers and dreams of becoming the Hokage, the leader of his village. The story is told in two parts: the first is set in Naruto's pre-teen years (volumes 1–27), and the second in his teens (volumes 28–72). The series is based on two one-shot manga by Kishimoto: Karakuri (1995), which earned Kishimoto an honorable mention in Shueisha's monthly Hop Step Award the following year, and Naruto (1997).

Naruto was serialized in Shueisha's shōnen manga magazine Weekly Shōnen Jump from September 1999 to November 2014, with its 700 chapters collected in 72 tankōbon volumes. Viz Media licensed the manga for North American production and serialized Naruto in their digital Weekly Shonen Jump magazine. The manga was adapted into two anime television series by Pierrot and Aniplex, which ran from October 2002 to March 2017 on TV Tokyo. Pierrot also produced 11 animated films and 12 original video animations (OVAs). The franchise additionally includes light novels, video games, and trading cards. The story continues in Boruto, where Naruto's son Boruto Uzumaki creates his own ninja path as opposed to following his father's.

Naruto is one of the best-selling manga series of all time, having sold over 250 million copies in 46 countries. It has become one of Viz Media's best-selling manga series; their English translations of the volumes have appeared on USA Today and The New York Times's bestseller list several times, and the seventh volume won a Quill Award in 2006. Naruto has been praised for its character development, storylines, and action sequences, though some felt the latter slowed the story down. Critics noted that the manga, which contains coming-of-age themes, often incorporates cultural references to Japanese mythology and Confucianism.
    """

    summary_template = """
    given the information {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them
    """
    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template
    )

    # llm #gemma3:270m
    llm = ChatOllama(temperature=0, model="gemma3:270m")
    # chain - o/p will be the response from the LLM based on the summary_prompt_templateand it will be a i/p to llm
    chain = summary_prompt_template | llm # LCEL - LangChain Expression Language for chaining prompt template and LLM
    response = chain.invoke(input={"information": information})
    print(response.content)


if __name__ == "__main__":
    main()
