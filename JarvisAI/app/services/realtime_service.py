from typing import List,Optional,Iterator,Tuple,Any
from tavily import TavilyClient
import logging
import os
import time
from app.services.groq_service import GroqService,escape_curly_braces,AllGroqApisFailedError
from app.services.vector_store import VectorStoreService
from app.utils.time_info import get_time_information #
from app.utils.retry import with_retry
from config import JARVIS_SYSTEM_PROMPT,REALTIME_CHAT_ADDENDUM,GROQ_API_KEYS,GROQ_MODEL,INTENT_CLASSIFY_MODEL,GENERAL_CHAT_ADDENDUM
from langchain_core.prompts import ChatPromptTemplate,  MessagesPlaceholder #
from langchain_core.messages import HumanMessage, AIMessage #

logger = logging.getLogger("J.A.R.V.I.S")
GROQ_REQUEST_TIMEOUT_FAST = 15  # seconds

_QUERY_EXTRACTION_PROMPT = (
    "You are a search query optimizer. Convert the user's message into a clean, focused "
    "web search query (max 10 words). Rules:\n"
    "- Remove filler words (you know, like, something, can you, tell me, search)\n"
    "- Add specifics: dates (today, 2026), event names, full names\n"
    "- For sports: include league name, team names, 'live score today'\n"
    "- For people: include full name + what user wants to know\n"
    "- Resolve references (him, that, it) from conversation history\n"
    "Output ONLY the search query. Nothing else."
)
class RealtimeGroqService(GroqService):
    def __init__(self, vector_store_service: VectorStoreService):
        super().__init__(vector_store_service)
        
        tavily_api_key = os.getenv("TAVILY_API_KEY"," ")
        
        if tavily_api_key:
            self .tavily_client = TavilyClient(api_key=tavily_api_key)
            logger.info("Tavily client initialized for real-time streaming") 
        else:
            self.tavily_client = None
            logger.info("TAVILY_API_KEY not set, real-time streaming will be disabled")
            
        if GROQ_API_KEYS:
            from langchain_groq import ChatGroq
            self._fast_llm = ChatGroq(
                groq_api_key=GROQ_API_KEYS[0],
                model_name=INTENT_CLASSIFY_MODEL,
                temperature=0.0,
                request_timeout=GROQ_REQUEST_TIMEOUT_FAST,
                max_tokens=50,
                
            )
            
        else:
            self._fast_llm = None
            
            
    def _extract_search_query(
        self,question: str,chat_history:Optional[List[tuple]] = None
                
    ) -> str:
        if not self._fast_llm:
            return question
        
        q=question.strip()
        q_lower = q.lower()
        
        has_filler = any(p in q_lower for p in (
            "it", "him", "her", "that", "this", "those", "these",
            "you know", "like", "something", "can you", "tell me", "search","please"
        ))
            
        if len(q) <= 30 and not has_filler:
            return q
        
        try:
            t0 = time.perf_counter()
            history_context = ""
            if chat_history:
                recent = chat_history[-3:]
                parts = []
                
                for h,a in recent:
                    parts.append(f"User: {h[:200]}")
                    parts.append(f"Assistant: {a[:200]}")
                history_context = "\n".join(parts)
                
            if history_context:
                full_prompt = (
                    f"(_QUERY_EXTRACTION_PROMPT)\n\n"
                    f"Recent conversation:\n{history_context}\n\n"
                    f"User's latest message: {question}\n\n"
                    f"Search query:"
                )
                
            else:
                full_prompt =(
                    f"(_QUERY_EXTRACTION_PROMPT)\n\n"
                    f"User's message: {question}\n\n"
                    f"Search query:"
                )
                
            response = self._fast_llm(full_prompt)
            extracted = response.content.strip('"').strip("'")
            
            if extracted and 3 <= len(extracted) <= 200:
                logger.info(
                   "[REALTIME] Query extraction: '%s' -> '%s' (%.3fs)",
                   question[:80], extracted[:80], time.perf_counter() - t0,
                )
                return extracted

            logger.warning("[REALTIME] Query extraction returned unusable result, using raw question")
            return question

        except Exception as e:
            logger.warning("[REALTIME] Query extraction failed (%s), using raw question", e)
            return question
                
            
    def search_tavily(self,query: str,num_results: int=5) -> str:
        if not self.tavily_client:
            logger.warning("Tavily client not initialized, cannot perform search")
            return("", None)
        if not query or not str(query).strip():
            return("", None)

        try:
            t0 = time.perf_counter()
            response= with_retry(
                lambda: self.tavily_client.search(
                    query=query,
                    search_depth="fast",
                    max_results=num_results,
                    include_raw_content=False,
                    include_answer=True,
                ),
                max_retries=3,
                initial_delay=1.0,
                
            )
            
            results = response.get("results",[])
            ai_answer = response.get("answer","")
            
            if not results and not ai_answer:
                logger.warning("No Tavily search results for query: %s ",query)
                return ("", None)
            
            payload: Optional[dict] = {
                "query": query,
                "answer": ai_answer,
                "results": [
                    {
                       "title":r.get("title","No Title"),
                       "content":(r.get("content") or "")[:300],
                       "url": r.get("url",""),
                       "score":round(float(r.get("score",0)),2),
                    }
                    for r in results [:num_results]
            
                    
                ],
            }
            
            
            parts = [f"=== WEB SEARCH RESULTS FOR '{query}' ===\n"]
            
            if ai_answer:
                parts.append(f"AI-Synthesized Answer:\n{ai_answer}\n")
                
            if results:
                parts.append("INDIVIDUAL SOURCES:")
            
            
           # formatted_results = f"Search results for '{query}':\n[start]\n"
            
            for i , result in enumerate(results[:num_results], 1):
                title = result.get('title','No Title')
                content = result.get('content','No description')
                url = result.get('url','No URL')
                score =  result.get('score',0)
                parts.append(f"\n[Source{i}] (relevance:{score:.2f})")
                parts.append(f"Title: {title}")
                
                if content:
                    parts.append(f"Content: {content}")
                    
                if url:
                    parts.append(f"URL: {url}")
                    
            parts.append("\n === END SEARCH RESULTS ===")
            formatted = "\n".join(parts)
            
            logger.info(
                "[TAVILY] %d results, AI answer: %s, fomatted:%d chars (%.3fs)",
                len(results),"yes" if ai_answer else "no",
                len(formatted), time.perf_counter() - t0,
            )
            return (formatted,payload)
        except Exception as e:
            logger.error("Error performing Tavily search: %s", e)
            return ("", None)
        
        
    def get_response(self,question: str, chat_history: Optional[List[tuple]] = None,key_start_index: int = 0) -> str:
            
        try:
            search_query = self._extract_search_query(question, chat_history)
            logger.info("[REALTIME] Search Travily for: %s", search_query)
            formatted_results,_ = self. search_tavily(search_query, num_results=5)
            
            if formatted_results:
                logger.info("[REALTIME] Tavily returned results (lenght: %d chars)", len(formatted_results))
            
            else:
                logger.warning("[REALTIME] Tavily returned no results for: %s", search_query)
                
            extra_parts = [escape_curly_braces(formatted_results)] if formatted_results else None
            prompt,messages = self._build_prompt_and_messages(
                question, chat_history,
                extra_system_parts=extra_parts,
                mode_addendum=REALTIME_CHAT_ADDENDUM,
            )
            
            t0= time.perf_counter()
            response_context = self._invoke_llm(prompt, messages, question, key_start_index=key_start_index)
            logger.info("[TIMING] groq_api: %.3fs", time.perf_counter() - t0)
            logger.info("[RESPONSE] Realtime chat | Length: %d chars | preview: %.120s", len(response_context), response_context,)
#           context=""
            
#            try:
 #               retriever = self.vector_store_service.get_retriever(k=10)
  #             context = "\n".join([doc.page_content for doc in context_docs]) if context_docs else ""
   #         except Exception as retrieval_error:
    #            logger.warning(f"Vector store retrieval failed: {retrieval_error}")
     #              
      #      time_info = get_time_information()
       #     system_message = JARVIS_SYSTEM_PROMPT + f"\n\nCurrent time: {time_info}"
            
        #    if search_results:
         #       escaped_search_results = escape_curly_braces(search_results)
          #      system_message += f"\n\nRecent search results:\n{escaped_search_results}"
                
           
           # if context:
           #     escaped_context = escape_curly_braces(context)
            #    system_message += f"\n\nRelevant context from your learning data and past conversations:\n{escaped_context}"
                
           # prompt = ChatPromptTemplate.from_messages([
            #    ("system", system_message),
             #   MessagesPlaceholder("history"),
              #  ("human", "{question}"),
            #])
            #messages =[]
 #           if chat_history:
  #              for human_msg, ai_msg in chat_history:
   #                 messages.append(HumanMessage(content=human_msg))
    #                messages.append(AIMessage(content=ai_msg))
            
     #       response_context = self._invoke_llm(prompt, messages,question)
      #      logger.info(f"Real-time response generated for: {question}") 
            return response_context
        except Exception as e:
            logger.error(f"Error in realtime Get_response: {e}",exc_info=True)
            raise
    def prefetch_web_search(
        self, question: str, chat_history: Optional[List[tuple]] = None
    ) -> Tuple[str, Optional[dict]]:
        
        try:
            t0= time.perf_counter()
            search_query = self._extract_search_query(question, chat_history)
            logger.info("[REALTIME] Pre-fetch: Extracted search query: '%s' in %.3fs", search_query[:60], time.perf_counter() - t0)
            formatted_results, payload = self.search_tavily(search_query, num_results=5)
            
            if formatted_results:
                logger.info("[REALTIME] pre-fetch: Tavily returned %d chars in %.3fs total",
                            len(formatted_results), time.perf_counter() - t0)
            return (formatted_results or "", payload)
        except Exception as e:
            logger.warning("[REALTIME] pre-fetch_web_search failed: %s", e)
            return ("", None)
        
    def stream_response(self, question, chat_history: Optional[List[tuple]] = None, key_start_index = 0)-> Iterator[Any]:
        
        try:
            yield {"_activity": {"event": "extracting_query", "message": "Extracting search query from user input..."}}
            search_query = self._extract_search_query(question, chat_history)
            logger.info("[REALTIME] Searching Tavily for: %s", search_query)
            yield {"_activity": {"event": "searching_web","query": search_query, "message": f"Performing web search for: '{search_query}'..."}}
            formatted_results, payload = self.search_tavily(search_query, num_results=5)
            num_results = len(payload.get("results", [])) if payload else 0
            
            if formatted_results:
                logger.info("[REALTIME] Tavily returned results (length: %d chars)", len(formatted_results))
                yield {"_activity": {"event": "search_completed", "message": f"Search completed: {num_results} results, {len(formatted_results)} chars of context"}}
            else:
                logger.warning("[REALTIME] Tavily returned no results for: %s", search_query)
                yield {"_activity": {"event": "search_completed", "message": "No search results found"}}

            if payload:
                yield {"_search_results": payload}

            extra_parts = [escape_curly_braces(formatted_results)] if formatted_results else None
            prompt, messages = self._build_prompt_and_messages(
                question, chat_history,
                extra_system_parts=extra_parts,
                mode_addendum=REALTIME_CHAT_ADDENDUM,
            )
            yield from self._stream_llm(prompt, messages, question, key_start_index=key_start_index)
            logger.info("[REALTIME] Stream completed for: %s", search_query)

        except AllGroqApisFailedError:
            raise

        except Exception as e:
            logger.error("Error in realtime stream_response: %s", e, exc_info=True)
            raise
    
    
    def stream_response_with_prefetched(
      self,
      question: str,
      chat_history: Optional[List[tuple]] = None,
      formatted_results: Optional[str] = None,
      payload: Optional[dict] = None,
      key_start_index: int = 0,
    ) -> Iterator[Any]:

        try:
            extra_parts = [escape_curly_braces(formatted_results)] if formatted_results else None
            prompt, messages = self._build_prompt_and_messages(
                question, chat_history,
                extra_system_parts=extra_parts,
                mode_addendum=REALTIME_CHAT_ADDENDUM,
            )
            yield from self._stream_llm(prompt, messages, question, key_start_index=key_start_index)
            logger.info("[REALTIME] Stream completed (pre-fetched results)")

        except AllGroqApisFailedError:
            raise

        except Exception as e:
            logger.error("Error in stream_response_with_prefetched: %s", e, exc_info=True)
            raise
            
