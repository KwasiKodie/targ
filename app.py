import os 
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings 

from module_loader import load 

retrieval = load(
    "retrieval_runtime",
    "retrieval.py"
)

VectorRetriever = retrieval.VectorRetriever

from targ_pipeline import TARGPipeline 

from draft_generator import DraftGenerator 
from margin_uncertainty_scorer import MarginUncertaintyScorer 
from retrieval_gate import RetrievalGate 
from answer_generator import AnswerGenerator 
from transformers import AutoModelForCausalLM, AutoTokenizer
from threshold_calibrator import ThresholdCalibrator

def main():

    load_dotenv()

    MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
    )

    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/e5-base-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    root_path = os.getenv("DATA_PATH")
    folder_path = os.path.join(root_path, "rag/vector_stores/authoritative_faiss")

    vector_store = FAISS.load_local(
        folder_path=folder_path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )

    retriever = VectorRetriever(
        vector_store=vector_store,
    )

    calibration = ThresholdCalibrator.load(
        "threshold.json"
    )

    pipeline = TARGPipeline(
        draft_generator=DraftGenerator(
            model=model,
            tokenizer=tokenizer,
            prefix_length=20,
        ),
        scorer=MarginUncertaintyScorer(beta=3.0),
        gate=RetrievalGate(calibration=calibration),
        retriever=retriever,
        answer_generator=AnswerGenerator(
            model=model,
            tokenizer=tokenizer,
        ),
    )

    while True:
        query = input("\nQuestion: ").strip()

        if query.lower() in {"exit", "quit"}:
            break 

        result = pipeline.run(query)

        print(f"\nAnswer:\n{result.answer.generated_text}")
        print(f"Retrieved: {result.gate.retrieve}")
        for stage, seconds in result.timing.items():
            print(f"{stage:20}: {seconds:.3f} s")

if __name__ == "__main__":
    main()