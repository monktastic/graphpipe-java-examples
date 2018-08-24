package com.foo;

import com.oracle.graphpipe.NativeTensor;
import com.oracle.graphpipe.Remote;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.zoo.util.imagenet.ImageNetLabels;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastSubOp;
import org.nd4j.linalg.factory.Nd4j;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.IOException;
import java.io.InputStream;


public class Main {
    static final int WIDTH = 224;
    static final int HEIGHT = 224;
    // Mean BGR pixel values (to be subtracted).
    static final double[] BGR_MEANS = new double[]{103.939, 116.779, 123.68};

    private static NativeTensor loadImage(InputStream input) throws
            IOException {
        BufferedImage image = ImageIO.read(input);
        if (image == null) {
           throw new UnsupportedOperationException("Could not read image");
        }
        if (image.getType() != BufferedImage.TYPE_3BYTE_BGR) {
            throw new UnsupportedOperationException();
        }
        if (image.getWidth() != WIDTH || image.getHeight() != HEIGHT) {
            BufferedImage outputImage = new BufferedImage(WIDTH, HEIGHT, image.getType());
            Graphics2D g2d = outputImage.createGraphics();
            g2d.drawImage(image, 0, 0, WIDTH, HEIGHT, null);
            g2d.dispose();
            image = outputImage;
        }

        byte[] pixels = ((DataBufferByte)image.getRaster().getDataBuffer()).getData();
        float[] fpixels = new float[pixels.length];
        for (int i = 0; i < pixels.length; i++) {
            fpixels[i] += (float)(pixels[i] & 0xFF) - BGR_MEANS[i % 3];
        }
        return NativeTensor.fromFlatArray(fpixels, new long[]{1, 
                HEIGHT, WIDTH, 3});
    }

    private static NativeTensor loadImageDL4J(InputStream input) 
            throws IOException {
        NativeImageLoader loader = new NativeImageLoader(WIDTH, HEIGHT, 3);
        INDArray image = loader.asMatrix(input);
        // We could use the DL4J VGG preprocessor, but there's a minor bug 
        // and it also flips BGR to RGB (which we don't want).
        // https://github.com/deeplearning4j/deeplearning4j/issues/6252
//        DataNormalization scaler = new VGG16ImagePreProcessor();
//        scaler.transform(image);
//        image = RGB2BGR(image);
        Nd4j.getExecutioner().execAndReturn(
                new BroadcastSubOp(image.dup(), Nd4j.create(BGR_MEANS), image, 1));

        // Channels-first -> channels-last.
        image = image.permute(0, 2, 3, 1);
        return NativeTensor.fromINDArray(image);
    }

    // Takes a 1 x W x H x 3 array and swaps the R/B channels.
    public static INDArray RGB2BGR(INDArray ary) {
        // Swap RGB into first dim, because putSlice() only allows dim 
        // (unlike slice(), which takes a dim).
        ary = ary.permute(3, 1, 2, 0);
        // Copy the R channel because it's about to be overwritten.
        INDArray red = ary.slice(0).dup();
        // Overwrite with B channel.
        ary.putSlice(0, ary.slice(2));
        // Write saved R channel to B.
        ary.putSlice(2, red);
        // Permute dimensions back.
        return ary.permute(3, 1, 2, 0);
    }
   
    static void printPreds(NativeTensor nt, ImageNetLabels inl) throws IOException {
        NativeTensor result = Remote.Execute("http://127.0.0.1:9001", nt);
        INDArray preds = result.toINDArray();
        String predStr = inl.decodePredictions(preds);
        System.out.println(predStr);
    }
   
    public static void main(String[] args) throws IOException {
        InputStream input = Main.class.getResourceAsStream("/dog_wide.jpg");
        ImageNetLabels inl = new ImageNetLabels();

        NativeTensor nt2 = loadImageDL4J(input);
        printPreds(nt2, inl);
        
        NativeTensor nt1 = loadImage(input);
        printPreds(nt1, inl);
    }
}
