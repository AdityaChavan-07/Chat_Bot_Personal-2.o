from typing import List, Optional,Iterator
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage,AIMessage
import logging
import time
from config import GROQ_API_KEYS, GROQ_MODEL, JARVIS_SYSTEM_PROMPT,GENERAL_CHAT_ADDENDUM
from app.services.vector_store import VectorStoreService
from app.utils.time_info import get_time_information
from app.utils.retry import with_retry

logger = logging.getLogger("J.A.R.V.I.S")
GROQ_REQUEST_TIMEOUT = 60  # seconds

ALL_APIS_FAILED_MESSAGE =(
    "I'M Unable to process your request right now because all of my language model APIs are currently unavailable. This could be due to high demand or technical issues on the provider's side. Please try again later, and I'll do my best to assist you as soon as I'm back online."
)

class AllGroqApisFailedError(Exception):
 """ Raised when all Groq API keys have been tried and failed to get a response.
 
 This is a custom exception so the API layer can catch it specifically and return a 503 service unavailable status, indicating that the issue is with the external service rather than the client's request.
 
 USAGE IN THE API LAYER:
 try:
        response = groq_service.get_response(question, history)
    except AllGroqApisFailedError:
        return JSONResponse(status_code=503, content={"detail": str(e)})
    """
 pass

def escape_curly_braces(text: str) -> str:
    """ Escape curly braces in the text to prevent formatting issues """
    
    if not text:
        return text
    return text.replace("{", "{{").replace("}", "}}")

_REPEAT_WINDOW =100
_REPEAT_THRESHOLD = 3
_REPEAT_CHECK_INTERVAL = 200  # seconds

def _detect_repetition_loop(text: str) -> bool:
    if len(text) < _REPEAT_WINDOW * _REPEAT_THRESHOLD:
        return False
    
    phrase = text[-_REPEAT_WINDOW:]
    return text.count(phrase) >= _REPEAT_THRESHOLD

def _truncate_at_repetition(text: str) -> str:

    if len(text) < _REPEAT_WINDOW * _REPEAT_THRESHOLD:
        return text

    phrase = text[-_REPEAT_WINDOW:]
    if text.count(phrase) < _REPEAT_THRESHOLD:
        return text

    first = text.find(phrase)
    second = text.find(phrase, first + 1)

    if second > first:
        return text[:second].rstrip()
    return text    

def _is_rate_limit_error(exc: BaseException) -> bool:
    """ Check if the error is a rate limit error based on its message """
    message = str(exc).lower()
    return "429" in str(exc) or "rate limit" in message or "token per day " in message

def _log_timing(label: str, elapsed: float, extra : str=""):
    """ Log timing information in a consistent format """
    msg = f"[TIMING] {label}: {elapsed:.3f}s"
    if extra:
        msg += f"({extra})"
    logger.info(msg)

def _mask_api_key(key: str) -> str:
    """ Mask the API key for logging purposes, showing only the last 4 characters """
    if not key or len(key) <= 12:
        return "***masked***"
    return  f"{key[:8]}... {key[-4:]}"

class GroqService:
    
    #_shared_key_index = 0
    #_lock= None
    
    def __init__(self, vector_store_service: VectorStoreService):
        
        if not GROQ_API_KEYS:
            raise ValueError("No GROQ API keys provided. Please set the GROQ_API_KEYS environment variable.")
        
        self.llms = [
            ChatGroq(
                groq_api_key=key,
                model_name=GROQ_MODEL,  
                temperature= 0.6,
                max_tokens= 2048,
                request_timeout=GROQ_REQUEST_TIMEOUT,
                model_kwargs={"frequency_penalty": 0.3},

            )
            for key in GROQ_API_KEYS
        ]
        self.vector_store_service = vector_store_service
        logger .info(f"Initialized GroqService with {len(GROQ_API_KEYS)}API keys(s) (primary-first fallback)")
        
    def _invoke_llm(
        self,
        prompt: ChatPromptTemplate,
        messages: list,
        question: str,
        key_start_index: int = 0,
    ) -> str:
        """ Internal method to invoke the LLM with the given prompt and messages, handling rate limits and errors """
        
        n= len(self.llms)
        last_exc = None
        keys_tried = []
        for j in range(n):
            i = (key_start_index + j) % n
            keys_tried.append(i)
            masked_keys = _mask_api_key(GROQ_API_KEYS[i])
            logger.info(f"Trying API key #{i + 1}/{n} : {masked_keys}")
            
            def _invoke_with_key():
                chain = prompt | self.llms[i]
                return chain.invoke({"history": messages, "question": question})
            try:
                response = with_retry(
                    _invoke_with_key,
                    max_retries=2,
                    initial_delay=0.5,
                )
                if i > 0:
                    logger.info(f"Fallback successful : API key #{i + 1}/{n} succeeded: {masked_keys}")
                text = response.content
                return text
            
            except Exception as e:
                last_exc =e
                if _is_rate_limit_error(e):
                    logger.warning(f"API key #{i + 1}/{n} : {masked_keys} hit rate limit. Trying next key if available...")
                else:
                    logger.warning(f"API key #{i + 1}/{n} failed: {masked_keys} - {str(e)[:100]}")
                if i < n - 1:
                    logger.info("Trying next key due to error...")
                    continue
                break
        masked_all = ",".join([_mask_api_key(GROQ_API_KEYS[j]) for j in keys_tried])
        logger.error(f"All {n} API key(s) failed. Tried keys: {masked_all}")
        raise AllGroqApisFailedError(ALL_APIS_FAILED_MESSAGE) from last_exc
      
       #start_i = GroqService._shared_key_index % n
        #current_key_index = GroqService._shared_key_index
        #GroqService._shared_key_index += 1
        
       # masked_keys = _mask_api_key(GROQ_API_KEYS[start_i])
        #logger.info(f"Using API key #{start_i + 1}/{n} (round - robin index: {current_key_index}) : {masked_keys}")
        
        #last_exc = None
        #keys_tried = []
        #for j in range(n):
            #i = (start_i + j) % n
            #keys_tried.append(i)
            #try:
                #chain = prompt | self.llms[i]
                #response = chain.invoke({"history": messages, "question": question})
                #if j > 0:
                 #   masked_success_key = _mask_api_key(GROQ_API_KEYS[i])
                  #  logger.info(f"fallback successful : API key #{i + 1}/{n} : {masked_success_key}")
                #return response.content
            #except Exception as e:
             #   last_exc = e
              #  masked_failed_key = _mask_api_key(GROQ_API_KEYS[i])
               # if _is_rate_limit_error(e):
                #    logger.warning(f"API key #{i + 1}/{n} : {masked_failed_key} hit rate limit. Trying next key if available...")
                #else:
                 #   logger.warning(f"API key #{i + 1}/{n} : {masked_failed_key} - {str(e)[:100]}")
                    
                #if n == 1:
                 #   raise Exception(f"Error getting response from Groq:{str(e)}") from e
                #continue
                
        #masked_all_keys = ",".join([_mask_api_key(GROQ_API_KEYS[i]) for i in keys_tried])
        #logger.error(f"ALL API keys failed.. Tried keys: {masked_all_keys}")
        #raise Exception(f"All API keys failed. Last error: {str(last_exc)}") from last_exc
        
        
    def _stream_llm(
        self,
        prompt: ChatPromptTemplate,
        messages: list,
        question: str,
        key_start_index: int = 0,
        ) -> Iterator[str]:
        """ Internal method to invoke the LLM in streaming mode, yielding chunks of the response as they arrive """
        
        n = len(self.llms)
        last_exc = None
        
        for j in range(n):
            i = (key_start_index + j) % n
            masked_key =  _mask_api_key(GROQ_API_KEYS[i])
            logger.info(f"Trying API key #{i + 1}/{n} for streaming: {masked_key}")
            try:
                chain = prompt | self.llms[i]
                chunk_count = 0
                first_chunk_time = None
                stream_start = time.perf_counter()
                accumulated = ""
                last_chunk_len = 0
                repetition_stopped = False
                
                for chunk in chain.stream({"history": messages, "question": question}):
                    content =""
                    if hasattr(chunk, "content"):
                        content = chunk.content or ""
                    elif isinstance(chunk, dict) and "content" in chunk:
                        content = chunk.get("content","") or ""

                    if isinstance(content, str) and content:
                        if first_chunk_time is None:
                            first_chunk_time = time.perf_counter() - stream_start
                            _log_timing("first_chunk", first_chunk_time)
                        chunk_count += 1
                        accumulated += content

                        if len(accumulated) - last_chunk_len > _REPEAT_CHECK_INTERVAL:
                            last_chunk_len = len(accumulated)

                        logger.debug(f"[STREAM-DEBUG] Chunk {chunk_count}: {len(content)} chars, accumulated: {len(accumulated)} chars")
                        yield content

                total_stream = time.perf_counter() - stream_start
                suffix = ",TRUNCATED-REPETITION" if repetition_stopped else ""
                _log_timing("groq_stream_total", total_stream, f"chunks: {chunk_count}{suffix}")
                logger.info(f"[STREAM-COMPLETE] Streamed {chunk_count} chunks, {len(accumulated)} total chars")
                if i> 0 and chunk_count > 0:
                    logger.info(f"Fallback successful: API key #{i+1}/{n} streamed :{masked_key}")
                return
            except Exception as e:
                last_exc = e
                if _is_rate_limit_error(e):
                    logger.warning(f"API key #{i + 1}/{n} : {masked_key} hit rate limit during streaming. Trying next key if available...")
                else:
                    logger.warning(f"API key #{i + 1}/{n} : {masked_key} failed during streaming - {str(e)[:100]}")
                if i < n - 1:
                    logger.info("Trying next key due to error...")
                    continue
                break
        logger.error(f"All {n} API Key(s) failed during stream.")
        raise AllGroqApisFailedError(ALL_APIS_FAILED_MESSAGE) from last_exc
        
        
    def _build_prompt_and_messages(
       self,
       question: str,
       chat_history: Optional[List[tuple]] = None,
       extra_system_parts: Optional[List[str]] = None,
       mode_addendum: str = "",
       
    ) -> tuple: 
        context = ""
        context_sources = []
        t0 = time.perf_counter()
        try:
            retriever = self.vector_store_service.get_retriever(k=5)
            context_docs = retriever.invoke(question)
            if context_docs:
              context = "\n".join([doc.page_content for doc in context_docs])
              context_sources = [doc.metadata.get("source", "unknown") for doc in context_docs]
              logger.info("[CONTEXT] Retrived %d  chunks from source: %s", len(context_docs),context_docs)
            else:
               logger.info("[CONTEXT] NO relevant chunks found for query")
        except Exception as retrieval_err:
            logger.warning("Vector store retrieval failed, using empty context: %s", retrieval_err)
        finally:
            _log_timing("vector_db", time.perf_counter() - t0)
            
        time_info = get_time_information()
        system_message = JARVIS_SYSTEM_PROMPT + f"\n\nCurrent time info: {time_info}"
        if context:
            system_message += f"\n\nRelevant information from your knowledge base:\n{escape_curly_braces(context)}"
        if extra_system_parts:
            system_message += "\n\n" + "\n".join(extra_system_parts)
        if mode_addendum:
                    system_message += f"\n\n{mode_addendum}"
        prompt = ChatPromptTemplate.from_messages([
                ("system", system_message),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ])
        messages = []
        if chat_history:
            for human_msg, ai_msg in chat_history:
                messages.append(HumanMessage(content=human_msg))
                messages.append(AIMessage(content=ai_msg))
                    
        logger.info(f"[prompt] System message length: %d chars | History pairs: %d | Question: %.100s", len(system_message), len(chat_history) if chat_history else 0, question)
        return prompt, messages
    
                    
       
        
        
    def get_response(
        self,
        question: str,
        chat_history: Optional[List[tuple]] = None,
        key_start_index: int = 0,
    ) -> str:
        """ Get a response from the LLM based on the question and chat history """
        try:
            prompt, messages = self._build_prompt_and_messages(question, chat_history, mode_addendum=GENERAL_CHAT_ADDENDUM)
            t0 = time.perf_counter()
            result = self._invoke_llm(prompt, messages, question)
            _log_timing("groq_api", time.perf_counter() - t0)
            logger.info(f"[RESPONSE] General chat | Length: %d chars | preview: %.120s",len(result), result)
            return result
        except AllGroqApisFailedError:
            raise
        except Exception as e:
            raise Exception(f"Error getting response from Groq: {str(e)}") from e
            

            #context = ""
        #    try:
         #     retriever= self.vector_store_service.get_retriever(k=10)
          #    context_docs = retriever.invoke(question)
           #   context = "\n".join([doc.page_content for doc in context_docs]) if context_docs else""
          #  except Exception as retrieval_err:
           #     logger.warning("vector store retrieval failed , using empty context: %s", retrieval_err) 
            
           # time_info = get_time_information()
           # system_message = JARVIS_SYSTEM_PROMPT + f"\n\nCurrent time info: {time_info}"  
           # if context:
            #    system_message += f"\n\nRelevant information from your knowledge base:\n{escape_curly_braces(context)}"    
        
            #prompt =ChatPromptTemplate.from_messages([
             #   ("system", system_message),
              #  MessagesPlaceholder(variable_name="history"),
               # ("human", "{question}"),
            
           #])    
        
           # messages = []
           # if chat_history:
            #    for human_msg, ai_msg in chat_history:
             #       messages.append(HumanMessage(content=human_msg))
              #      messages.append(AIMessage(content=ai_msg))
   
           # return self._invoke_llm(prompt, messages, question)
        
        # Catch any errors that occur during response generation and wrap them with a clear message.
        # This keeps the exception handling within the method scope.
        
       # except Exception as e:
        #   raise Exception(f"Error getting response from Groq: {str(e)}") from e
    
    def stream_response(
        self,
        question: str,
        chat_history: Optional[List[tuple]] = None,
        key_start_index: int = 0,
    ) -> Iterator[str]:
        """ Get a response from the LLM in streaming mode, yielding chunks as they arrive """
        try:
           prompt, messages = self._build_prompt_and_messages(question, chat_history, mode_addendum=GENERAL_CHAT_ADDENDUM)
           yield{"_activity":{"event":"context_retrieved","message":"Retrieved relevent context from knowledge base"}}
           yield from self._stream_llm(prompt, messages, question)
        except AllGroqApisFailedError:
            raise
        except Exception as e:
           raise Exception(f"Error getting streaming response from Groq: {str(e)}") from e
