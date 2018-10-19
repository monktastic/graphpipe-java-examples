/*
** Copyright © 2018, Oracle and/or its affiliates. All rights reserved.
** Licensed under the Universal Permissive License v 1.0 as shown at http://oss.oracle.com/licenses/upl.
*/
package com.foo;

import com.oracle.graphpipe.NativeTensor;
import com.oracle.graphpipe.Remote;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.zoo.util.imagenet.ImageNetLabels;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastSubOp;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.io.InputStream;


/**
 * Call a graphpipe server running VGG16.
 */
public class Main {
    private static final String URL = "http://127.0.0.1:9001";
    private static final int WIDTH = 224;
    private static final int HEIGHT = 224;
    // Mean BGR pixel values (to be subtracted).
    private static final double[] BGR_MEANS =
            new double[]{103.939, 116.779, 123.68};

    private static NativeTensor loadImage(InputStream input)
            throws IOException {
        NativeImageLoader loader = new NativeImageLoader(HEIGHT, WIDTH, 3);
        INDArray image = loader.asMatrix(input);
        // NOTE: We could use the DL4J VGG16ImagePreProcessor, but there's a 
        // minor bug, and it also flips BGR to RGB (which we don't want).
        // https://github.com/deeplearning4j/deeplearning4j/issues/6252
        Nd4j.getExecutioner().execAndReturn(
                new BroadcastSubOp(
                        image.dup(), Nd4j.create(BGR_MEANS), image, 1));

        // Channels-first -> channels-last.
        image = image.permute(0, 2, 3, 1);
        return NativeTensor.fromINDArray(image);
    }

    private static void printPreds(NativeTensor nt, ImageNetLabels inl)
            throws IOException {
        NativeTensor result = Remote.Execute(URL, nt);
        INDArray preds = result.toINDArray();
        String predStr = inl.decodePredictions(preds);
        System.out.println(predStr);
    }

    public static void main(String[] args) throws IOException {
        InputStream input = Main.class.getResourceAsStream("/dog_wide.jpg");
        ImageNetLabels inl = new ImageNetLabels();
        NativeTensor nt2 = loadImage(input);
        printPreds(nt2, inl);
    }
}
