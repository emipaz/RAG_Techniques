"""
Módulo de funciones para el procesamiento de documentos PDF y técnicas RAG.

Este módulo contiene funciones para:
- Procesamiento de documentos PDF
- Codificación de texto en vectores
- Recuperación de contexto
- Generación de respuestas basadas en contexto
- Búsqueda BM25
- Manejo de reintentos con backoff exponencial
"""

import textwrap
import fitz  # PyMuPDF
import random
import asyncio
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from enum import Enum

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.pydantic_v1 import BaseModel, Field
# from langchain import PromptTemplate
from langchain_core.prompts import PromptTemplate
from openai import RateLimitError
from rank_bm25 import BM25Okapi


def reemplazar_tabs_por_espacios(lista_documentos: List) -> List:
    """
    Reemplaza todos los caracteres de tabulación ('\t') con espacios en el contenido de cada documento.

    Esta función es útil para limpiar documentos que contienen tabulaciones irregulares
    que pueden afectar el procesamiento posterior del texto.

    Args:
        lista_documentos (List): Lista de objetos documento, cada uno con un atributo 'page_content'.

    Returns:
        List: La lista modificada de documentos con caracteres de tabulación reemplazados por espacios.

    Example:
        >>> docs = [Mock(page_content="Texto\tcon\ttabs")]
        >>> resultado = reemplazar_tabs_por_espacios(docs)
        >>> print(docs[0].page_content)
        "Texto con tabs"
    """
    for documento in lista_documentos:
        documento.page_content = documento.page_content.replace('\t', ' ')
    return lista_documentos


def envolver_texto(texto: str, ancho: int = 120) -> str:
    """
    Envuelve el texto de entrada al ancho especificado.

    Utiliza la biblioteca textwrap para dividir líneas largas de texto
    en múltiples líneas que no excedan el ancho especificado.

    Args:
        texto (str): El texto de entrada a envolver.
        ancho (int, optional): El ancho al cual envolver el texto. Por defecto 120.

    Returns:
        str: El texto envuelto con saltos de línea apropiados.

    Example:
        >>> texto_largo = "Este es un texto muy largo que necesita ser envuelto"
        >>> resultado = envolver_texto(texto_largo, ancho=20)
        >>> print(resultado)
        "Este es un texto muy\nlargo que necesita ser\nenvuelto"
    """
    return textwrap.fill(texto, width=ancho)


def codificar_pdf(ruta: str, tamaño_chunk: int = 1000, solapamiento_chunk: int = 200) -> FAISS:
    """
    Codifica un documento PDF en un almacén vectorial usando embeddings de OpenAI.

    Esta función carga un PDF, lo divide en chunks manejables, limpia el texto
    y crea un almacén vectorial FAISS para búsquedas de similitud.

    Args:
        ruta (str): La ruta al archivo PDF.
        tamaño_chunk (int, optional): El tamaño deseado de cada chunk de texto. Por defecto 1000.
        solapamiento_chunk (int, optional): La cantidad de solapamiento entre chunks consecutivos. Por defecto 200.

    Returns:
        FAISS: Un almacén vectorial FAISS que contiene el contenido codificado del documento.

    Raises:
        FileNotFoundError: Si no se encuentra el archivo PDF.
        ValueError: Si los parámetros de chunk no son válidos.

    Example:
        >>> vectorstore = codificar_pdf("documento.pdf", tamaño_chunk=500)
        >>> # El vectorstore ahora puede usarse para búsquedas de similitud
    """
    # Cargar documento PDF
    cargador   = PyPDFLoader(ruta)
    documentos = cargador.load()

    # Dividir documentos en chunks
    divisor_texto = RecursiveCharacterTextSplitter(
        chunk_size      = tamaño_chunk, 
        chunk_overlap   = solapamiento_chunk, 
        length_function = len
    )
    
    textos         = divisor_texto.split_documents(documentos)
    textos_limpios = reemplazar_tabs_por_espacios(textos)

    # Crear embeddings y almacén vectorial
    embeddings        = OpenAIEmbeddings()
    almacen_vectorial = FAISS.from_documents(textos_limpios, embeddings)

    return almacen_vectorial


def codificar_desde_cadena(contenido: str, tamaño_chunk: int = 1000, solapamiento_chunk: int = 200) -> FAISS:
    """
    Codifica una cadena de texto en un almacén vectorial usando embeddings de OpenAI.

    Divide el contenido de texto en chunks manejables y crea un almacén vectorial
    para búsquedas de similitud semántica.

    Args:
        contenido (str): El contenido de texto a ser codificado.
        tamaño_chunk (int, optional): El tamaño de cada chunk de texto. Por defecto 1000.
        solapamiento_chunk (int, optional): El solapamiento entre chunks. Por defecto 200.

    Returns:
        FAISS: Un almacén vectorial que contiene el contenido codificado.

    Raises:
        ValueError: Si el contenido de entrada no es válido o los parámetros son incorrectos.
        RuntimeError: Si hay un error durante el proceso de codificación.

    Example:
        >>> contenido = "Este es un texto largo para codificar..."
        >>> vectorstore = codificar_desde_cadena(contenido)
        >>> # Usar vectorstore para búsquedas
    """
    # Validación de entrada
    if not isinstance(contenido, str) or not contenido.strip():
        raise ValueError("El contenido debe ser una cadena no vacía.")

    if not isinstance(tamaño_chunk, int) or tamaño_chunk <= 0:
        raise ValueError("tamaño_chunk debe ser un entero positivo.")

    if not isinstance(solapamiento_chunk, int) or solapamiento_chunk < 0:
        raise ValueError("solapamiento_chunk debe ser un entero no negativo.")

    try:
        # Dividir el contenido en chunks
        divisor_texto = RecursiveCharacterTextSplitter(
            chunk_size         = tamaño_chunk,
            chunk_overlap      = solapamiento_chunk,
            length_function    = len,
            is_separator_regex = False,
        )

        chunks = divisor_texto.create_documents([contenido])

        # Asignar metadatos a cada chunk
        for i, chunk in enumerate(chunks):
            chunk.metadata = {"chunk_id": i, "source": "string_input"}

        # Generar embeddings y crear el almacén vectorial
        embeddings        = OpenAIEmbeddings()
        almacen_vectorial = FAISS.from_documents(chunks, embeddings)

        return almacen_vectorial

    except Exception as e:
        raise RuntimeError(f"Error durante la codificación: {str(e)}")


def recuperar_contexto_por_pregunta(pregunta: str, recuperador_consulta_chunks) -> List[str]:
    """
    Recupera contexto relevante para una pregunta dada usando el recuperador de chunks.

    Args:
        pregunta (str): La pregunta para la cual recuperar contexto.
        recuperador_consulta_chunks: El recuperador de chunks configurado.

    Returns:
        List[str]: Lista con el contenido de los documentos relevantes.

    Example:
        >>> contexto = recuperar_contexto_por_pregunta("¿Qué es IA?", recuperador)
        >>> print(len(contexto))
        5
    """
    # Recuperar documentos relevantes para la pregunta dada
    documentos = recuperador_consulta_chunks.get_relevant_documents(pregunta)
    
    # Extraer contenido de los documentos
    contexto = [doc.page_content for doc in documentos]
    
    return contexto


class RespuestaPreguntaDesdeContexto(BaseModel):
    """
    Modelo para generar una respuesta a una consulta basada en un contexto dado.
    
    Attributes:
        respuesta_basada_en_contenido (str): La respuesta generada basada en el contexto.
    """
    respuesta_basada_en_contenido: str = Field(
        description="Genera una respuesta a una consulta basada en un contexto dado."
    )


def crear_cadena_respuesta_pregunta_desde_contexto(llm):
    """
    Crea una cadena de procesamiento para responder preguntas basadas en contexto.

    Args:
        llm: El modelo de lenguaje a utilizar.

    Returns:
        Una cadena configurada para generar respuestas estructuradas.

    Example:
        >>> cadena = crear_cadena_respuesta_pregunta_desde_contexto(mi_llm)
        >>> respuesta = cadena.invoke({"pregunta": "¿Qué es esto?", "contexto": "..."})
    """
    # Inicializar el modelo de lenguaje
    llm_respuesta_pregunta_contexto = llm

    # Definir la plantilla de prompt para razonamiento en cadena
    plantilla_prompt_respuesta_pregunta = """
    Para la siguiente pregunta, proporciona una respuesta concisa pero suficiente basada ÚNICAMENTE en el contexto proporcionado:
    
    Contexto:
    {contexto}
    
    Pregunta:
    {pregunta}
    
    Respuesta:
    """

    # Crear objeto PromptTemplate
    prompt_respuesta_pregunta_contexto = PromptTemplate(
        template        = plantilla_prompt_respuesta_pregunta,
        input_variables = ["contexto", "pregunta"],
    )

    # Crear cadena combinando prompt y modelo de lenguaje
    cadena_respuesta_pregunta_contexto = prompt_respuesta_pregunta_contexto | llm_respuesta_pregunta_contexto.with_structured_output(
        RespuestaPreguntaDesdeContexto
    )
    
    return cadena_respuesta_pregunta_contexto


def responder_pregunta_desde_contexto(pregunta: str, contexto: str, cadena_respuesta_pregunta_contexto) -> Dict[str, Any]:
    """
    Responde una pregunta usando el contexto dado invocando una cadena de razonamiento.

    Args:
        pregunta (str): La pregunta a ser respondida.
        contexto (str): El contexto a ser usado para responder la pregunta.
        cadena_respuesta_pregunta_contexto: La cadena configurada para generar respuestas.

    Returns:
        Dict[str, Any]: Diccionario que contiene la respuesta, contexto y pregunta.

    Example:
        >>> resultado = responder_pregunta_desde_contexto(
        ...     "¿Qué es Python?", 
        ...     "Python es un lenguaje de programación...",
        ...     str
        ... )
        >>> print(resultado["respuesta"])
    """
    datos_entrada = {
        "pregunta": pregunta,
        "contexto": contexto
    }
    print("Respondiendo la pregunta desde el contexto recuperado...")

    salida    = cadena_respuesta_pregunta_contexto.invoke(datos_entrada)
    respuesta = salida.respuesta_basada_en_contenido
    
    return {
        "respuesta": respuesta, 
        "contexto" : contexto, 
        "pregunta" : pregunta
    }


def mostrar_contexto(contexto: List[str]) -> None:
    """
    Muestra el contenido de la lista de contexto proporcionada.

    Args:
        contexto (List[str]): Lista de elementos de contexto a mostrar.

    Example:
        >>> contextos = ["Contexto 1", "Contexto 2"]
        >>> mostrar_contexto(contextos)
        === Contexto 1 ===
        Contexto 1
        === Contexto 2 ===
        Contexto 2
    """
    for i, c in enumerate(contexto):
        print(f"=== Contexto {i + 1} ===")
        print(c)
        print()


def leer_pdf_a_cadena(ruta: str) -> str:
    """
    Lee un documento PDF desde la ruta especificada y devuelve su contenido como cadena.

    Args:
        ruta (str): La ruta del archivo al documento PDF.

    Returns:
        str: El contenido de texto concatenado de todas las páginas del documento PDF.

    Raises:
        FileNotFoundError: Si no se encuentra el archivo PDF.
        Exception: Si hay un error al leer el PDF.

    Example:
        >>> contenido = leer_pdf_a_cadena("documento.pdf")
        >>> print(len(contenido))
        15000
    """
    try:
        # Abrir el documento PDF ubicado en la ruta especificada
        documento  = fitz.open(ruta)
        contenido = ""
        
        # Iterar sobre cada página del documento
        for num_pagina in range(len(documento)):
            pagina     = documento.load_page(num_pagina)
            contenido += pagina.get_text()
            
        documento.close()
        return contenido
        
    except FileNotFoundError:
        raise FileNotFoundError(f"No se encontró el archivo PDF en la ruta: {ruta}")
    except Exception as e:
        raise Exception(f"Error al leer el PDF: {str(e)}")


def recuperacion_bm25(bm25: BM25Okapi, textos_limpios: List[str], consulta: str, k: int = 5) -> List[str]:
    """
    Realiza recuperación BM25 y devuelve los top k chunks de texto limpios.

    Args:
        bm25 (BM25Okapi): Índice BM25 pre-computado.
        textos_limpios (List[str]): Lista de chunks de texto limpios correspondientes al índice BM25.
        consulta (str): La cadena de consulta.
        k (int, optional): El número de chunks de texto a recuperar. Por defecto 5.

    Returns:
        List[str]: Los top k chunks de texto limpios basados en puntuaciones BM25.

    Example:
        >>> resultados = recuperacion_bm25(indice_bm25, textos, "machine learning", k=3)
        >>> print(len(resultados))
        3
    """
    # Tokenizar la consulta
    tokens_consulta = consulta.split()

    # Obtener puntuaciones BM25 para la consulta
    puntuaciones_bm25 = bm25.get_scores(tokens_consulta)

    # Obtener los índices de las top k puntuaciones
    indices_top_k = np.argsort(puntuaciones_bm25)[::-1][:k]

    # Recuperar los top k chunks de texto limpios
    textos_top_k = [textos_limpios[i] for i in indices_top_k]

    return textos_top_k


async def backoff_exponencial(intento: int) -> None:
    """
    Implementa backoff exponencial con jitter.
    
    Args:
        intento (int): El número de intento de reintento actual.
        
    Espera por un período de tiempo antes de reintentar la operación.
    El tiempo de espera se calcula como (2^intento) + una fracción aleatoria de segundo.
    
    Example:
        >>> await backoff_exponencial(1)  # Espera ~2-3 segundos
        >>> await backoff_exponencial(3)  # Espera ~8-9 segundos
    """
    # Calcular el tiempo de espera con backoff exponencial y jitter
    tiempo_espera = (2 ** intento) + random.uniform(0, 1)
    print(f"Límite de tasa alcanzado. Reintentando en {tiempo_espera:.2f} segundos...")

    # Dormir asincrónicamente por el tiempo calculado
    await asyncio.sleep(tiempo_espera)


async def reintentar_con_backoff_exponencial(corutina, max_reintentos: int = 5):
    """
    Reintenta una corutina usando backoff exponencial al encontrar un RateLimitError.
    
    Args:
        corutina: La corutina a ser ejecutada.
        max_reintentos (int, optional): El número máximo de intentos de reintento. Por defecto 5.
        
    Returns:
        El resultado de la corutina si es exitosa.
        
    Raises:
        Exception: La última excepción encontrada si todos los intentos de reintento fallan.

    Example:
        >>> resultado = await reintentar_con_backoff_exponencial(mi_corutina(), max_reintentos=3)
    """
    for intento in range(max_reintentos):
        try:
            return await corutina
        except RateLimitError as e:
            if intento == max_reintentos - 1:
                raise e
            await backoff_exponencial(intento)
        except Exception as e:
            raise e

    # Si se alcanzan los reintentos máximos sin éxito, lanzar excepción
    raise Exception("Máximo de reintentos alcanzado")


# Clase enum que representa diferentes proveedores de embeddings
class ProveedorEmbeddings(Enum):
    """
    Enumeración de proveedores de embeddings disponibles.
    """
    OPENAI = "openai"
    COHERE = "cohere"
    AMAZON_BEDROCK = "bedrock"


# Clase enum que representa diferentes proveedores de modelos
class ProveedorModelo(Enum):
    """
    Enumeración de proveedores de modelos de lenguaje disponibles.
    """
    OPENAI = "openai"
    GROQ = "groq"
    ANTHROPIC = "anthropic"
    AMAZON_BEDROCK = "bedrock"


def obtener_proveedor_embeddings_langchain(proveedor: ProveedorEmbeddings, id_modelo: str = None):
    """
    Devuelve un proveedor de embeddings basado en el proveedor y ID de modelo especificados.

    Args:
        proveedor (ProveedorEmbeddings): El proveedor de embeddings a usar.
        id_modelo (str, optional): El ID específico del modelo de embeddings a usar.

    Returns:
        Una instancia del proveedor de embeddings de LangChain.

    Raises:
        ValueError: Si el proveedor especificado no está soportado.

    Example:
        >>> embeddings = obtener_proveedor_embeddings_langchain(
        ...     ProveedorEmbeddings.OPENAI, 
        ...     "text-embedding-ada-002"
        ... )
    """
    if proveedor == ProveedorEmbeddings.OPENAI:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=id_modelo) if id_modelo else OpenAIEmbeddings()
        
    elif proveedor == ProveedorEmbeddings.COHERE:
        from langchain_cohere import CohereEmbeddings
        return CohereEmbeddings(model=id_modelo) if id_modelo else CohereEmbeddings()
        
    elif proveedor == ProveedorEmbeddings.AMAZON_BEDROCK:
        from langchain_aws import BedrockEmbeddings
        return BedrockEmbeddings(model_id=id_modelo) if id_modelo else BedrockEmbeddings()
        
    else:
        raise ValueError(f"Proveedor de embeddings no soportado: {proveedor}")

