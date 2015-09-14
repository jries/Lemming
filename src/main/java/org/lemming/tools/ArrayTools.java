package org.lemming.tools;

import java.util.ArrayList;

import static java.lang.Math.*;
import static java.lang.Math.max;
import static java.lang.Math.pow;

/**
 * Created with IntelliJ IDEA.
 * User: Ricardo Henriques <paxcalpt@gmail.com>
 * Date: 14/09/15
 * Time: 16:17
 */
public class ArrayTools {

    public static void normalize(float[] numbers, float normalizeTo) {
        if (numbers.length == 1) {
            numbers[0] = normalizeTo;
            return;
        }
        float vMax = numbers[0];
        float vMin = numbers[0];
        for(int i=0; i<numbers.length; i++) {
            vMax = max(numbers[i], vMax);
            vMin = min(numbers[i], vMin);
        }
        if (vMax == 0 && vMin == 0) return;
        vMax -= vMin;
        for(int i=0; i<numbers.length; i++) {
            numbers[i] = (numbers[i]-vMin) * normalizeTo / vMax;
        }
    }

    public static void normalizeOneMinusOne(float[] numbers) {
        normalize(numbers, 2);
        for(int i=0; i<numbers.length; i++) {
            numbers[i] -= 1;
        }
    }

    public static void normalizeDontSubtractMin(float[] numbers, float normalizeTo) {
        float vMax = getMaxValue(numbers)[1];
        if (vMax == 0) return;
        for(int i=0; i<numbers.length; i++) {
            numbers[i] = numbers[i] * normalizeTo / vMax;
        }
    }

    public static void normalizeIntegratedIntensity(float[] numbers, float normalizeTo) {
        float vSum = getAbsAverageValue(numbers) * numbers.length;
        for(int i=0; i<numbers.length; i++) numbers[i] *= normalizeTo / vSum;
    }

    /**
     *
     * @param numbers
     * @return float[] {position of max, value of max}
     */
    public static float [] getMaxValue(float[] numbers){
        float [] value = new float [2];
        value[0] = 0;
        value[1] = numbers[0];
        for(int i=0; i<numbers.length; i++){
            if(numbers[i] > value[1]){
                value[0] = i;
                value[1] = numbers[i];
            }
        }
        return value;
    }

    public static float [] getMinValue(float[] numbers){
        float [] value = new float [2];
        value[0] = 0;
        value[1] = numbers[0];
        for(int i=0; i<numbers.length; i++){
            if(numbers[i] < value[1]){
                value[0] = i;
                value[1] = numbers[i];
            }
        }
        return value;
    }

    public static float getMaxMinusMinValue(float[] numbers){
        float maxV = numbers[0];
        float minV = numbers[0];
        float v;
        for(int i=0; i<numbers.length; i++){
            v = numbers[i];
            maxV = max(v, maxV);
            minV = min(v, minV);
        }
        return maxV-minV;
    }

    public static float getMaxMinusMeanValue(float[] numbers){
        float maxV = numbers[0];
        float meanV = 0;
        float v;
        for(int i=0; i<numbers.length; i++){
            v = numbers[i];
            maxV = max(v, maxV);
            meanV += v;
        }
        return maxV-(meanV / numbers.length);
    }

    public static float getAverageValue(float[] numbers){
        float v = 0;
        for(int i=0; i<numbers.length; i++){
            v += numbers[i] / numbers.length;
        }
        return v;
    }

    public static float getSumValue(float[] numbers){
        float v = 0;
        for(int i=0; i<numbers.length; i++) v += numbers[i];
        return v;
    }

    public static float getStandardDeviationValue(float[] numbers){
        float v = 0;
        float average = getAverageValue(numbers);
        for(int i=0; i<numbers.length; i++){
            v += Math.pow(numbers[i]-average,2);
        }
        v = (float) sqrt(v)/numbers.length;
        return v;
    }

    public static float getAbsAverageValue(float[] numbers) {
        float v = 0;
        for(int i=0; i<numbers.length; i++){
            v += abs(numbers[i]) / numbers.length;
        }
        return v;
    }

    /**
     * Makes convert any value smaller than v in the array to v
     * @param numbers
     * @param v
     * @return
     */
    public static void setMinValue(float[] numbers, float v){
        for(int i=0; i < numbers.length;i++){
            if(numbers[i] < v) numbers[i] = v;
        }
    }

    public static void addWithReplace(float[] numbers, float v){
        for(int i=0; i < numbers.length;i++) numbers[i] += v;
    }

    public static float[] multiply(float[] numbers, float v){
        float[] numbers_ = numbers.clone();
        for(int i=0; i < numbers.length;i++) numbers_[i] *= v;
        return numbers_;
    }

    public static float[] add(float[] numbers1, float[] numbers2){
        assert (numbers1.length==numbers2.length);
        float[] numbers = new float[numbers1.length];
        for(int i=0; i < numbers.length;i++)
            numbers[i] = (float) numbers1[i]+numbers2[i];
        return numbers;
    }

    public static float add(float[] numbers) {
        float v = 0;
        for(int i=0; i < numbers.length;i++) v+=numbers[i];
        return v;
    }

    public static void add(float[] numbers, float value) {
        for(int i=0; i < numbers.length;i++) numbers[i] += value;
    }

    public static void add(float[] numbers, double value) {
        for(int i=0; i < numbers.length;i++) numbers[i] += value;
    }

    public static float[] l2norm(float[] numbers1, float[] numbers2){
        assert (numbers1.length==numbers2.length);
        float[] numbers = new float[numbers1.length];
        for(int i=0; i < numbers.length;i++)
            numbers[i] = (float) Math.sqrt(numbers1[i]*numbers1[i]+numbers2[i]*numbers2[i]);
        return numbers;
    }

    ///////////////////////////
    // Array type conversion //
    ///////////////////////////

    public static float[] arrayListToFloatArray(ArrayList<Float> array) {
        float[] values = new float[array.size()];
        for (int n=0; n<array.size(); n++) values[n] = array.get(n);
        return values;
    }

    public static int[] encodeFloatArrayIntoInt(float [] floatArr, int precision){
        int [] intArr = new int[floatArr.length];
        for (int n=0; n<intArr.length; n++) {
            intArr[n] = (int) Math.round(floatArr[n]*pow(10, precision));
        }
        return intArr;
    }

    public static float[] decodeFloatArrayFromInt(int [] intArr, int precision){
        float [] floatArr = new float[intArr.length];
        for (int n=0; n<intArr.length; n++) {
            floatArr[n] = (float) (intArr[n]/pow(10, precision));
        }
        return floatArr;
    }

    public static float[] intArray2floatArray(int [] intArr){
        float [] floatArr = new float[intArr.length];
        for (int n=0; n<intArr.length; n++) {
            floatArr[n] = (float) intArr[n];
        }
        return floatArr;
    }

    public static float[] doubleToFloat(double[] doubleArray){
        //Takes a 1D double array and converts to a 1D float array
        //Nils Gustafsson

        float[] floatArray = new float[doubleArray.length];
        for (int i = 0 ; i < doubleArray.length; i++)
            floatArray[i] = (float) doubleArray[i];
        return floatArray;
    }

    public static double[] floatToDouble(float[] floatArray){
        //Takes a 1D float array and converts to a 1D double array
        //Nils Gustafsson

        double[] doubleArray = new double[floatArray.length];
        for (int i = 0 ; i < floatArray.length; i++)
            doubleArray[i] = (double) floatArray[i];
        return doubleArray;
    }

    public static int[] ArrayListInteger2int(ArrayList<Integer> integerArray){
        int[] intArray = new int[integerArray.size()];
        for (int i = 0 ; i < intArray.length; i++)
            intArray[i] = integerArray.get(i);
        return intArray;
    }

    public static float[] ArrayListFloat2float(ArrayList<Float> FloatArray){
        float[] floatArray = new float[FloatArray.size()];
        for (int i = 0 ; i < floatArray.length; i++)
            floatArray[i] = FloatArray.get(i);
        return floatArray;
    }

    ///////////////////////
    // Initialize arrays //
    ///////////////////////

    public static float[] initializeAndValueFill(int size, float value){
        float[] array = new float[size];
        for (int n=0;n<array.length;n++) array[n]=value;
        return array;
    }

    public static int[] initializeAndValueFill(int size, int value){
        int[] array = new int[size];
        for (int n=0;n<array.length;n++) array[n]=value;
        return array;
    }

    public static float[] initializeFloatAndGrowthFill(int size, float startingValue, float increment){
        float[] array = new float[size];
        for (int n=0;n<array.length;n++) array[n]=startingValue+n*increment;
        return array;
    }

    ///////////////////////
    // Filter arrays //
    ///////////////////////

    public static float[] smoothArray(float[] numbers, int radius){
        float[] arraySmooth = new float[numbers.length];
        int nIterations = 2*radius+1;

        for (int n=0;n<numbers.length;n++) {
            for (int i=-radius;i<=radius;i++){
                arraySmooth[n]+=numbers[Math.min(max(n + i, 0), numbers.length-1)]/nIterations;
            }

        }
        return arraySmooth;
    }
}
