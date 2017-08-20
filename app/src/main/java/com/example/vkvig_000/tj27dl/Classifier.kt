package com.example.vkvig_000.tj27dl

import android.graphics.Bitmap
import android.graphics.RectF

interface Classifier {

    class Recognition(

            val id: String?,

            val title: String?,

            val confidence: Float?,

            private var location: RectF?) {

        override fun toString(): String {
            var resultString = ""
            if (id != null) {
                resultString += "[$id] "
            }

            if (title != null) {
                resultString += title + " "
            }

            if (confidence != null) {
                resultString += String.format("(%.1f%%) ", confidence * 100.0f)
            }

            if (location != null) {
                resultString += location!!.toString() + " "
            }

            return resultString.trim { it <= ' ' }
        }
    }


    fun recognizeImage(bitmap: Bitmap): List<Recognition>

    fun enableStatLogging(debug: Boolean)

    val statString: String

    fun close()
}

