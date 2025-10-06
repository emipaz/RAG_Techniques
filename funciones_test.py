"""
Tests unitarios para el módulo Funciones.py

Este archivo contiene pruebas exhaustivas para todas las funciones
del módulo Funciones.py, incluyendo casos edge y manejo de errores.

Fecha: 5 de octubre de 2025
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, mock_open
import pytest
import asyncio
import tempfile
import os
from typing import List
import numpy as np

# Importación del módulo a testear
from Funciones import (
    reemplazar_tabs_por_espacios,
    envolver_texto,
    codificar_pdf,
    codificar_desde_cadena,
    recuperar_contexto_por_pregunta,
    RespuestaPreguntaDesdeContexto,
    crear_cadena_respuesta_pregunta_desde_contexto,
    responder_pregunta_desde_contexto,
    mostrar_contexto,
    leer_pdf_a_cadena,
    recuperacion_bm25,
    backoff_exponencial,
    reintentar_con_backoff_exponencial,
    ProveedorEmbeddings,
    ProveedorModelo,
    obtener_proveedor_embeddings_langchain
)


class TestFuncionesProcesamiento(unittest.TestCase):
    """
    Clase de pruebas para las funciones de procesamiento de texto básico.
    """

    def test_reemplazar_tabs_por_espacios_exitoso(self):
        """
        Prueba que la función reemplaza correctamente las tabulaciones con espacios.
        """
        # Crear documentos simulados con tabulaciones
        mock_doc1 = Mock()
        mock_doc1.page_content = "Texto\tcon\ttabulaciones"
        mock_doc2 = Mock()
        mock_doc2.page_content = "Otro\ttexto\tcon\ttabs"
        
        documentos = [mock_doc1, mock_doc2]
        
        # Ejecutar función
        resultado = reemplazar_tabs_por_espacios(documentos)
        
        # Verificar que las tabulaciones fueron reemplazadas
        self.assertEqual(mock_doc1.page_content, "Texto con tabulaciones")
        self.assertEqual(mock_doc2.page_content, "Otro texto con tabs")
        self.assertEqual(len(resultado), 2)
        self.assertIs(resultado, documentos)  # Debe retornar la misma lista

    def test_reemplazar_tabs_por_espacios_lista_vacia(self):
        """
        Prueba que la función maneja correctamente una lista vacía.
        """
        resultado = reemplazar_tabs_por_espacios([])
        self.assertEqual(resultado, [])

    def test_reemplazar_tabs_por_espacios_sin_tabs(self):
        """
        Prueba que la función no modifica texto sin tabulaciones.
        """
        mock_doc = Mock()
        mock_doc.page_content = "Texto sin tabulaciones"
        documentos = [mock_doc]
        
        resultado = reemplazar_tabs_por_espacios(documentos)
        
        self.assertEqual(mock_doc.page_content, "Texto sin tabulaciones")

    def test_envolver_texto_ancho_por_defecto(self):
        """
        Prueba que la función envuelve texto usando el ancho por defecto.
        """
        texto_largo = "Este es un texto muy largo " * 10  # Crear texto largo
        resultado = envolver_texto(texto_largo)
        
        self.assertIsInstance(resultado, str)
        lineas = resultado.split('\n')
        # Verificar que ninguna línea exceda el ancho por defecto (120)
        for linea in lineas:
            self.assertLessEqual(len(linea), 120)

    def test_envolver_texto_ancho_personalizado(self):
        """
        Prueba que la función envuelve texto con ancho personalizado.
        """
        texto = "Este es un texto que necesita ser envuelto en líneas cortas"
        ancho = 20
        resultado = envolver_texto(texto, ancho=ancho)
        
        lineas = resultado.split('\n')
        for linea in lineas:
            self.assertLessEqual(len(linea), ancho)

    def test_envolver_texto_corto(self):
        """
        Prueba que la función no modifica texto corto.
        """
        texto_corto = "Texto corto"
        resultado = envolver_texto(texto_corto, ancho=50)
        self.assertEqual(resultado, texto_corto)


class TestFuncionesPDF(unittest.TestCase):
    """
    Clase de pruebas para las funciones de procesamiento de PDF.
    """

    @patch('Funciones.PyPDFLoader')
    @patch('Funciones.OpenAIEmbeddings')
    @patch('Funciones.FAISS')
    @patch('Funciones.RecursiveCharacterTextSplitter')
    def test_codificar_pdf_exitoso(self, mock_splitter, mock_faiss, mock_embeddings, mock_loader):
        """
        Prueba que la función codifica correctamente un PDF.
        """
        # Configurar mocks
        mock_doc = Mock()
        mock_doc.page_content = "Contenido del documento PDF"
        
        mock_loader_instance = Mock()
        mock_loader_instance.load.return_value = [mock_doc]
        mock_loader.return_value = mock_loader_instance
        
        mock_splitter_instance = Mock()
        mock_splitter_instance.split_documents.return_value = [mock_doc]
        mock_splitter.return_value = mock_splitter_instance
        
        mock_vectorstore = Mock()
        mock_faiss.from_documents.return_value = mock_vectorstore
        
        # Ejecutar función
        resultado = codificar_pdf("ruta/test.pdf", tamaño_chunk=500, solapamiento_chunk=100)
        
        # Verificar llamadas a los mocks
        mock_loader.assert_called_once_with("ruta/test.pdf")
        mock_loader_instance.load.assert_called_once()
        mock_splitter.assert_called_once_with(
            chunk_size=500, chunk_overlap=100, length_function=len
        )
        mock_faiss.from_documents.assert_called_once()
        self.assertEqual(resultado, mock_vectorstore)

    def test_codificar_desde_cadena_exitoso(self):
        """
        Prueba que la función codifica correctamente una cadena de texto.
        """
        with patch('Funciones.RecursiveCharacterTextSplitter') as mock_splitter, \
             patch('Funciones.OpenAIEmbeddings') as mock_embeddings, \
             patch('Funciones.FAISS') as mock_faiss:
            
            # Configurar mocks
            mock_chunk = Mock()
            mock_chunk.metadata = {}
            
            mock_splitter_instance = Mock()
            mock_splitter_instance.create_documents.return_value = [mock_chunk]
            mock_splitter.return_value = mock_splitter_instance
            
            mock_vectorstore = Mock()
            mock_faiss.from_documents.return_value = mock_vectorstore
            
            # Ejecutar función
            contenido = "Este es un contenido de prueba para codificar"
            resultado = codificar_desde_cadena(contenido, tamaño_chunk=100, solapamiento_chunk=20)
            
            # Verificar resultado
            self.assertEqual(resultado, mock_vectorstore)
            self.assertEqual(mock_chunk.metadata["chunk_id"], 0)
            self.assertEqual(mock_chunk.metadata["source"], "string_input")

    def test_codificar_desde_cadena_contenido_vacio(self):
        """
        Prueba que la función lanza error con contenido vacío.
        """
        with self.assertRaises(ValueError) as context:
            codificar_desde_cadena("")
        
        self.assertIn("cadena no vacía", str(context.exception))

    def test_codificar_desde_cadena_parametros_invalidos(self):
        """
        Prueba que la función valida correctamente los parámetros.
        """
        contenido = "Contenido válido"
        
        # Tamaño de chunk inválido
        with self.assertRaises(ValueError):
            codificar_desde_cadena(contenido, tamaño_chunk=0)
        
        # Solapamiento negativo
        with self.assertRaises(ValueError):
            codificar_desde_cadena(contenido, solapamiento_chunk=-1)

    @patch('Funciones.fitz')
    def test_leer_pdf_a_cadena_exitoso(self, mock_fitz):
        """
        Prueba que la función lee correctamente un PDF.
        """
        # Configurar mock
        mock_doc = Mock()
        mock_pagina = Mock()
        mock_pagina.get_text.return_value = "Texto de la página "
        mock_doc.load_page.return_value = mock_pagina
        mock_doc.__len__.return_value = 2
        mock_fitz.open.return_value = mock_doc
        
        # Ejecutar función
        resultado = leer_pdf_a_cadena("test.pdf")
        
        # Verificar resultado
        self.assertEqual(resultado, "Texto de la página Texto de la página ")
        mock_fitz.open.assert_called_once_with("test.pdf")
        self.assertEqual(mock_doc.load_page.call_count, 2)
        mock_doc.close.assert_called_once()

    @patch('Funciones.fitz')
    def test_leer_pdf_a_cadena_archivo_no_encontrado(self, mock_fitz):
        """
        Prueba que la función maneja correctamente archivos no encontrados.
        """
        mock_fitz.open.side_effect = FileNotFoundError("Archivo no encontrado")
        
        with self.assertRaises(FileNotFoundError) as context:
            leer_pdf_a_cadena("archivo_inexistente.pdf")
        
        self.assertIn("archivo_inexistente.pdf", str(context.exception))


class TestFuncionesContexto(unittest.TestCase):
    """
    Clase de pruebas para las funciones de manejo de contexto y preguntas.
    """

    def test_recuperar_contexto_por_pregunta(self):
        """
        Prueba que la función recupera correctamente el contexto.
        """
        # Crear mock del recuperador
        mock_recuperador = Mock()
        mock_doc1 = Mock(page_content="Contexto 1")
        mock_doc2 = Mock(page_content="Contexto 2")
        mock_recuperador.get_relevant_documents.return_value = [mock_doc1, mock_doc2]
        
        # Ejecutar función
        resultado = recuperar_contexto_por_pregunta("¿Qué es IA?", mock_recuperador)
        
        # Verificar resultado
        self.assertEqual(resultado, ["Contexto 1", "Contexto 2"])
        mock_recuperador.get_relevant_documents.assert_called_once_with("¿Qué es IA?")

    def test_crear_cadena_respuesta_pregunta_desde_contexto(self):
        """
        Prueba que la función crea correctamente la cadena de procesamiento.
        """
        # Crear mock del LLM
        mock_llm = Mock()
        mock_llm.with_structured_output.return_value = Mock()
        
        # Ejecutar función
        resultado = crear_cadena_respuesta_pregunta_desde_contexto(mock_llm)
        
        # Verificar que se creó la cadena
        self.assertIsNotNone(resultado)
        mock_llm.with_structured_output.assert_called_once()

    def test_responder_pregunta_desde_contexto(self):
        """
        Prueba que la función responde correctamente preguntas.
        """
        # Crear mock de la cadena
        mock_respuesta = Mock()
        mock_respuesta.respuesta_basada_en_contenido = "Esta es la respuesta"
        
        mock_cadena = Mock()
        mock_cadena.invoke.return_value = mock_respuesta
        
        # Ejecutar función
        resultado = responder_pregunta_desde_contexto(
            "¿Qué es Python?", 
            "Python es un lenguaje de programación", 
            mock_cadena
        )
        
        # Verificar resultado
        self.assertEqual(resultado["respuesta"], "Esta es la respuesta")
        self.assertEqual(resultado["pregunta"], "¿Qué es Python?")
        self.assertEqual(resultado["contexto"], "Python es un lenguaje de programación")

    @patch('builtins.print')
    def test_mostrar_contexto(self, mock_print):
        """
        Prueba que la función muestra correctamente el contexto.
        """
        contextos = ["Primer contexto", "Segundo contexto"]
        
        mostrar_contexto(contextos)
        
        # Verificar que se llamó print con los valores esperados
        llamadas_print = mock_print.call_args_list
        self.assertTrue(any("=== Contexto 1 ===" in str(call) for call in llamadas_print))
        self.assertTrue(any("=== Contexto 2 ===" in str(call) for call in llamadas_print))


class TestFuncionesBM25(unittest.TestCase):
    """
    Clase de pruebas para las funciones de búsqueda BM25.
    """

    def test_recuperacion_bm25(self):
        """
        Prueba que la función BM25 recupera correctamente documentos.
        """
        # Crear mock de BM25
        mock_bm25 = Mock()
        mock_bm25.get_scores.return_value = np.array([0.1, 0.8, 0.3, 0.9, 0.2])
        
        textos = ["texto1", "texto2", "texto3", "texto4", "texto5"]
        
        # Ejecutar función
        resultado = recuperacion_bm25(mock_bm25, textos, "consulta test", k=3)
        
        # Verificar que retorna los top 3 basados en puntuaciones
        self.assertEqual(len(resultado), 3)
        self.assertIn("texto4", resultado)  # Puntuación más alta (0.9)
        self.assertIn("texto2", resultado)  # Segunda puntuación más alta (0.8)


class TestFuncionesAsync(unittest.TestCase):
    """
    Clase de pruebas para las funciones asíncronas.
    """

    @patch('Funciones.asyncio.sleep')
    @patch('Funciones.random.uniform')
    async def test_backoff_exponencial(self, mock_random, mock_sleep):
        """
        Prueba que la función de backoff exponencial calcula correctamente los tiempos.
        """
        mock_random.return_value = 0.5
        
        await backoff_exponencial(2)
        
        # Verificar que se calculó el tiempo correcto: 2^2 + 0.5 = 4.5
        mock_sleep.assert_called_once_with(4.5)

    async def test_reintentar_con_backoff_exponencial_exitoso(self):
        """
        Prueba que la función de reintento funciona cuando la operación es exitosa.
        """
        async def operacion_exitosa():
            return "éxito"
        
        resultado = await reintentar_con_backoff_exponencial(operacion_exitosa())
        self.assertEqual(resultado, "éxito")

    @patch('Funciones.backoff_exponencial')
    async def test_reintentar_con_backoff_exponencial_con_reintentos(self, mock_backoff):
        """
        Prueba que la función reintenta correctamente tras errores de rate limit.
        """
        from openai import RateLimitError
        
        intentos = 0
        async def operacion_con_fallo():
            nonlocal intentos
            intentos += 1
            if intentos < 3:
                raise RateLimitError("Rate limit", None, None)
            return "éxito tras reintentos"
        
        resultado = await reintentar_con_backoff_exponencial(operacion_con_fallo())
        self.assertEqual(resultado, "éxito tras reintentos")
        self.assertEqual(mock_backoff.call_count, 2)  # 2 reintentos


class TestEnumsYProveedores(unittest.TestCase):
    """
    Clase de pruebas para enums y funciones de proveedores.
    """

    def test_proveedor_embeddings_valores(self):
        """
        Prueba que el enum ProveedorEmbeddings tiene los valores correctos.
        """
        self.assertEqual(ProveedorEmbeddings.OPENAI.value, "openai")
        self.assertEqual(ProveedorEmbeddings.COHERE.value, "cohere")
        self.assertEqual(ProveedorEmbeddings.AMAZON_BEDROCK.value, "bedrock")

    def test_proveedor_modelo_valores(self):
        """
        Prueba que el enum ProveedorModelo tiene los valores correctos.
        """
        self.assertEqual(ProveedorModelo.OPENAI.value, "openai")
        self.assertEqual(ProveedorModelo.GROQ.value, "groq")
        self.assertEqual(ProveedorModelo.ANTHROPIC.value, "anthropic")
        self.assertEqual(ProveedorModelo.AMAZON_BEDROCK.value, "bedrock")

    @patch('Funciones.OpenAIEmbeddings')
    def test_obtener_proveedor_embeddings_langchain_openai(self, mock_openai):
        """
        Prueba que la función retorna correctamente el proveedor OpenAI.
        """
        obtener_proveedor_embeddings_langchain(ProveedorEmbeddings.OPENAI)
        mock_openai.assert_called_once()

    @patch('Funciones.CohereEmbeddings')
    def test_obtener_proveedor_embeddings_langchain_cohere(self, mock_cohere):
        """
        Prueba que la función retorna correctamente el proveedor Cohere.
        """
        with patch.dict('sys.modules', {'langchain_cohere': Mock()}):
            obtener_proveedor_embeddings_langchain(ProveedorEmbeddings.COHERE, "modelo-test")
            # Verificar que se llamó con el modelo específico
            mock_cohere.assert_called_once_with(model="modelo-test")

    def test_obtener_proveedor_embeddings_langchain_no_soportado(self):
        """
        Prueba que la función lanza error para proveedores no soportados.
        """
        with self.assertRaises(ValueError) as context:
            # Crear un valor enum inválido para probar
            proveedor_invalido = Mock()
            proveedor_invalido.value = "proveedor_inexistente"
            obtener_proveedor_embeddings_langchain(proveedor_invalido)
        
        self.assertIn("no soportado", str(context.exception))


class TestModelos(unittest.TestCase):
    """
    Clase de pruebas para los modelos Pydantic.
    """

    def test_respuesta_pregunta_desde_contexto_modelo(self):
        """
        Prueba que el modelo RespuestaPreguntaDesdeContexto funciona correctamente.
        """
        respuesta = RespuestaPreguntaDesdeContexto(
            respuesta_basada_en_contenido="Esta es una respuesta de prueba"
        )
        
        self.assertEqual(
            respuesta.respuesta_basada_en_contenido, 
            "Esta es una respuesta de prueba"
        )


if __name__ == '__main__':
    # Ejecutar todas las pruebas
    print("=== Ejecutando Tests Unitarios para Funciones.py ===")
    unittest.main(verbosity=2)