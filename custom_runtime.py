from mlserver import MLModel
from mlserver.types import InferenceRequest, InferenceResponse, ResponseOutput
from transformers import AutoTokenizer, AutoModelForCausalLM

class CustomHuggingFaceRuntime(MLModel):
    async def load(self) -> None:
        """Load the Hugging Face model and tokenizer from Hugging Face Hub."""
        model_id = self.settings.parameters.extra["model_id"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id).to("cpu")  # Ensure model is on CPU
        self.ready = True  # Mark the model as loaded

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        """Perform inference and return the generated text."""

        # Extract the input data
        prompt = payload.inputs[0].data[0]

        # Tokenize the prompt and move input tensors to CPU
        inputs = self.tokenizer(
            str(prompt), return_tensors="pt", truncation=True
        ).to("cpu")  # Ensures processing happens on CPU

        # Generate a response
        output = self.model.generate(
            input_ids=inputs["input_ids"],
            max_new_tokens=100,  # Generate up to 100 new tokens
            num_beams=5,     # Beam search width
            early_stopping=True,
            no_repeat_ngram_size=2,  # Avoid repetition
            temperature=0.7,         # Randomness control
            pad_token_id=self.tokenizer.pad_token_id  # Ensures padding is handled correctly
        )

        # Decode the generated text
        response_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

        # Create response output
        response_output = ResponseOutput(
            name="generated_text",
            shape=[1],
            datatype="BYTES",
            data=[response_text]
        )

        return InferenceResponse(model_name=self.name, outputs=[response_output])
