package com.example.vkvig_000.tj27dl

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.support.v7.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.widget.*
import kotlinx.android.synthetic.main.activity_main.*

import com.flurgle.camerakit.CameraListener
import com.flurgle.camerakit.CameraView


import java.util.concurrent.Executors
import org.jetbrains.anko.toast


class MainActivity : AppCompatActivity() {


    private val INPUT_SIZE = 224
    private val IMAGE_MEAN = 117
    private val IMAGE_STD = 1f
    private val INPUT_NAME = "input"
    private val OUTPUT_NAME = "output"

    private val TAG = "TFLog"

    private val MODEL_FILE = "file:///android_asset/tensorflow_inception_graph.pb"
    private val LABEL_FILE = "file:///android_asset/imagenet_comp_graph_label_strings.txt"

    private val executor = Executors.newSingleThreadExecutor()
    private var classifier: Classifier? = null
    private var imageViewResult: ImageView? = null
    private var cameraView: CameraView? = null
    private var assetManager: AssetManager? = null
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        cameraView = findViewById<ListView>(R.id.cameraView) as CameraView


        //cameraView!!.toggleFacing()

        cameraView!!.setCameraListener(object : CameraListener() {
            override fun onPictureTaken(picture: ByteArray?) {
                Log.i(TAG, "onPictureTaken")
                super.onPictureTaken(picture)
                Log.i(TAG, "Camera Listener loop")
                var bitmap = BitmapFactory.decodeByteArray(picture, 0, picture!!.size)
                Log.i(TAG, "decodeByteArray")
                bitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, false)
                Log.i(TAG, "createScaledBitmap")

                val results = classifier!!.recognizeImage(bitmap)

                toast("o/p: " + results.toString())
                Log.i(TAG, "Classified" + results.toString())
            }
        })
        initTensorFlowAndLoadModel()

        predict.setOnClickListener{
            cameraView!!.captureImage()
            Log.i(TAG, "Read Bitmap")
            toast("fw-prop..")
        }
    }

    override fun onResume() {
        super.onResume()
        cameraView!!.start()
    }

    override fun onPause() {
        cameraView!!.stop()
        super.onPause()
    }

    override fun onDestroy() {
        super.onDestroy()
        executor.execute { classifier!!.close() }
    }

    private fun initTensorFlowAndLoadModel() {
        executor.execute(Runnable {
            try {

                classifier = TensorFlowImageClassifier.create(
                        assets,
                        MODEL_FILE,
                        LABEL_FILE,
                        INPUT_SIZE,
                        IMAGE_MEAN,
                        IMAGE_STD,
                        INPUT_NAME,
                        OUTPUT_NAME)

            } catch (e: Exception) {
                throw RuntimeException("Error initializing TensorFlow!", e)
            }
        })
    }
}
