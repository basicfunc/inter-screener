import os
from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain


def analyze_text(model_name, API_KEY, transcription):
	os.environ['OPENAI_API_KEY'] = API_KEY

	tp = """
	As an interview analyzer, you are tasked with conducting a comprehensive analysis of a recent interview to evaluate the candidate's communication skills, problem-solving abilities, and overall suitability.
	In addition, you will assess their level of enthusiasm, adaptability, and cultural fit within the organization.
	Provide an in-depth examination of the interview, including specific examples that highlight the candidate's strengths and areas for improvement.
	Based on the assessment, you must provide recommendations to get better in future.

	Interview: {script}
	"""

	prompt = PromptTemplate(template=tp, input_variables=["script"])

	llm = OpenAI(model=model_name, temperature=0.9, verbose=True)

	llm_chain = LLMChain(prompt=prompt, llm=llm)

	with open('transcribe.txt') as s:
		script = s.read()

	result = llm_chain.run(script)

	return result