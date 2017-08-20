package com.example.vkvig_000.tj27dl

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.os.Trace
import android.util.Log

import org.tensorflow.contrib.android.TensorFlowInferenceInterface

import java.io.BufferedReader
import java.io.IOException
import java.io.InputStreamReader
import java.lang.RuntimeException
import java.util.ArrayList
import java.util.Comparator
import java.util.PriorityQueue
import java.util.Vector

class TensorFlowImageClassifier private constructor() : Classifier {

    private var inputName: String? = null
    private var outputName: String? = null
    private var inputSize: Int = 0
    private var imageMean: Int = 0
    private var imageStd: Float = 0.toFloat()

    private var labels = Vector<String>()
    private var intValues: IntArray? = null
    private var floatValues: FloatArray? = null
    private var outputs: FloatArray? = null
    private var outputNames: Array<String>? = null

    private var inferenceInterface: TensorFlowInferenceInterface? = null

    override fun recognizeImage(bitmap: Bitmap): List<Classifier.Recognition> {

        Trace.beginSection("recognizeImage")
        Trace.beginSection("preprocessBitmap")
        // Preprocess the image data from 0-255 int to normalized float based
        // on the provided parameters.
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        for (i in intValues!!.indices) {
            val `val` = intValues!![i]
            floatValues!![i * 3 + 0] = ((`val` shr 16 and 0xFF) - imageMean) / imageStd
            floatValues!![i * 3 + 1] = ((`val` shr 8 and 0xFF) - imageMean) / imageStd
            floatValues!![i * 3 + 2] = ((`val` and 0xFF) - imageMean) / imageStd
        }
        Trace.endSection()

        // Copy the input data into TensorFlow.
        Trace.beginSection("fillNodeFloat")
        inferenceInterface!!.fillNodeFloat(
                inputName!!, intArrayOf(1, inputSize, inputSize, 3), floatValues)
        Trace.endSection()

        // Run the inference call.
        Trace.beginSection("runInference")
        inferenceInterface!!.runInference(outputNames!!)
        Trace.endSection()

        // Copy the output Tensor back into the output array.
        Trace.beginSection("readNodeFloat")
        inferenceInterface!!.readNodeFloat(outputName, outputs)
        Trace.endSection()

        // Find the best classifications.
        val pq = PriorityQueue(
                3,
                Comparator<Classifier.Recognition> { lhs, rhs ->
                    // Intentionally reversed to put high confidence at the head of the queue.
                    java.lang.Float.compare(rhs.confidence!!, lhs.confidence!!)
                })
        for (i in outputs!!.indices) {
            if (outputs!![i] > THRESHOLD) {
                pq.add(
                        Classifier.Recognition(
                                "" + i, if (labels.size > i) labels[i] else "unknown", outputs!![i], null))
            }
        }
        val recognitions = ArrayList<Classifier.Recognition>()
        val recognitionsSize = Math.min(pq.size, MAX_RESULTS)
        for (i in 0..recognitionsSize - 1) {
            recognitions.add(pq.poll())
        }
        Trace.endSection() // "recognizeImage"
        return recognitions
    }

    override fun enableStatLogging(debug: Boolean) {
        inferenceInterface!!.enableStatLogging(debug)
    }

    override val statString: String
        get() = inferenceInterface!!.statString

    override fun close() {
        inferenceInterface!!.close()
    }

    companion object {

        private val TAG = "TFLog"

        // Only return this many results with at least this confidence.
        private val MAX_RESULTS = 3
        private val THRESHOLD = 0.1f


        @Throws(IOException::class)
        fun create(
                assetManager: AssetManager,
                modelFilename: String,
                labelFilename: String,
                inputSize: Int,
                imageMean: Int,
                imageStd: Float,
                inputName: String,
                outputName: String): Classifier {
            val c = TensorFlowImageClassifier()
            c.inputName = inputName
            c.outputName = outputName

            // Read the label names into memory.
            // TODO(andrewharp): make this handle non-assets.
            val actualFilename = labelFilename.split("file:///android_asset/".toRegex()).dropLastWhile({ it.isEmpty() }).toTypedArray()[1]
            Log.i(TAG, "Reading labels from: " + actualFilename)
            var br: BufferedReader? = null
            br = BufferedReader(InputStreamReader(assetManager.open(actualFilename)))
            Log.i(TAG, "Buffer read")
            var line: List<String>
            line = br.readLines()

            for (i in line) {
                //Log.i(TAG, "Label:" + i)
                c.labels.add(i)
            }

            Log.i(TAG, "Labels read")
            br.close()

            c.inferenceInterface = TensorFlowInferenceInterface()
            if (c.inferenceInterface!!.initializeTensorFlow(assetManager, modelFilename) != 0) {
                Log.i(TAG, "TF initialization failed")
                throw RuntimeException("TF initialization failed")

            }

            // The shape of the output is [N, NUM_CLASSES], where N is the batch size.
            val numClasses = c.inferenceInterface!!.graph().operation(outputName).output(0).shape().size(1).toInt()
            Log.i(TAG, "Read " + c.labels.size + " labels, output layer size is " + numClasses)

            // Ideally, inputSize could have been retrieved from the shape of the input operation.  Alas,
            // the placeholder node for input in the graphdef typically used does not specify a shape, so it
            // must be passed in as a parameter.
            c.inputSize = inputSize
            c.imageMean = imageMean
            c.imageStd = imageStd

            // Pre-allocate buffers.
            c.outputNames = arrayOf(outputName)
            c.intValues = IntArray(inputSize * inputSize)
            c.floatValues = FloatArray(inputSize * inputSize * 3)
            c.outputs = FloatArray(numClasses)

            return c
        }
    }
}
