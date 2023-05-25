from langchain import PromptTemplate, LLMChain

tp = """
Act as an interviewer, you are tasked with conducting a comprehensive analysis of a recent interview response to evaluate the candidate's communication skills, problem-solving abilities, and overall suitability.
In addition, you will assess their level of enthusiasm, adaptability, and cultural fit within the organization.
Identify the content what candidate is answering, and check whether he/she is correct on topic or not.

Use bullet points.
Be sharp.

Response: {script}

Answer: 
	1. Discuss candidate's strengths and weaknesses
	2. Discuss candidate's grammatical mistakes (if any).
	3. Provide description of context, speech figures, and Verbal fluency.
	4. Discuss areas for improvement.
	5. Write a short conclusion and give overall rating out of 10. 
"""


def analyze_text(model_name: str, API_KEY: str, text: str) -> str:
	from langchain.llms import OpenAI
	import os

	os.environ['OPENAI_API_KEY'] = API_KEY

	prompt = PromptTemplate(template=tp, input_variables=["script"])

	llm = OpenAI(model=model_name, temperature=0.9, verbose=True)

	llm_chain = LLMChain(prompt=prompt, llm=llm)

	script = text.strip()

	result = llm_chain.run(script)

	return result


def analyze_locally(model_path:str, backend:str, text:str) -> str:
	from langchain.llms import GPT4All
	from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

	prompt = PromptTemplate(template=tp, input_variables=["script"])

	callbacks = [StreamingStdOutCallbackHandler()]
	llm = GPT4All(model=model_path, callbacks=callbacks, verbose=False, backend=backend)

	llm_chain = LLMChain(prompt=prompt, llm=llm)

	script = text.strip()

	result = llm_chain.run(script)

	return result

def postProcess_output(output):
    lines = output.strip().splitlines()
    res = [line.strip() for line in lines]
    return "\n".join(res)