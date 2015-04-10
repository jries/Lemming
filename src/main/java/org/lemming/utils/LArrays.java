package org.lemming.utils;

import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

/**This Lemming-Arrays (LArrays) wrapper class is used for manipulating arrays 
 * (e.g., sorting, searching) and for performing statistical analysis (e.g., 
 * average, standard deviation). It implements all of the {@link java.util.Arrays}
 * methods, plus custom methods */
@SuppressWarnings("javadoc")
public class LArrays {
	
	private static String EMPTY_MESSAGE = "You are trying to analyse an empty array";
	
	/** If {@code condition} is true then raise a runtime exception 
	 * @param condition - the condition to test
	 * @param msg - the message to display */
	private static void check(boolean condition, String msg) {
		if (condition) throw new RuntimeException(msg);
	}

	/** Returns the average value in the array 
	 * 
	 * @param a - array to process
	 * @return value 
	 */
	public static double ave(byte[] a) {
		check(a.length==0, EMPTY_MESSAGE);
		double sum = 0.0;
		for (double value : a)
			sum += value;
		return sum/(double)a.length;
	}

	/** Returns the average value in the array
	 * 
	 * @param a - array to process
	 * @return value 
	 */
	public static double ave(short[] a) {
		check(a.length==0, EMPTY_MESSAGE);
		double sum = 0.0;
		for (double value : a)
			sum += value;
		return sum/(double)a.length;
	}
	
	/** Returns the average value in the array
	 * 
	 * @param a - array to process
	 * @return value 
	 */
	public static double ave(int[] a) {
		check(a.length==0, EMPTY_MESSAGE);
		double sum = 0.0;
		for (double value : a)
			sum += value;
		return sum/(double)a.length;
	}

	/** Returns the average value in the array
	 * 
	 * @param a - array to process
	 * @return value 
	 */
	public static double ave(long[] a) {
		check(a.length==0, EMPTY_MESSAGE);
		double sum = 0.0;
		for (double value : a)
			sum += value;
		return sum/(double)a.length;
	}

	/** Returns the average value in the array
	 * 
	 * @param a - array to process
	 * @return value 
	 */
	public static double ave(float[] a) {
		check(a.length==0, EMPTY_MESSAGE);
		double sum = 0.0;
		for (double value : a)
			sum += value;
		return sum/(double)a.length;
	}

	/** Returns the average value in the array
	 * 
	 * @param a - array to process
	 * @return value 
	 */
	public static double ave(double[] a) {
		check(a.length==0, EMPTY_MESSAGE);
		double sum = 0.0;
		for (double value : a)
			sum += value;
		return sum/(double)a.length;
	}
	
	/** Returns the maximum value in the array
	 * 
	 * @param a - array to process
	 * @return value 
	 */
	public static byte max(byte[] a) {
		check(a.length==0, EMPTY_MESSAGE);
		byte amax = Byte.MIN_VALUE;
		for (byte value : a)			
			if (value > amax)				
				amax = value;		
		return amax;
	}

	/** Returns the maximum value in the array
	 * 
	 * @param a - array to process
	 * @return value 
	 */
	public static short max(short[] a) {
		check(a.length==0, EMPTY_MESSAGE);
		short amax = Short.MIN_VALUE;
		for (short value : a)			
			if (value > amax)				
				amax = value;		
		return amax;
	}

	/** Returns the maximum value in the array
	 * 
	 * @param a - array to process
	 * @return value 
	 */
	public static int max(int[] a) {
		check(a.length==0, EMPTY_MESSAGE);
		int amax = Integer.MIN_VALUE;
		for (int value : a)			
			if (value > amax)				
				amax = value;		
		return amax;
	}

	/** Returns the maximum value in the array
	 * 
	 * @param a - array to process
	 * @return value 
	 */
	public static long max(long[] a) {
		check(a.length==0, EMPTY_MESSAGE);
		long amax = Long.MIN_VALUE;
		for (long value : a)			
			if (value > amax)				
				amax = value;		
		return amax;
	}

	/** Returns the maximum value in the array
	 * 
	 * @param a - array to process
	 * @return value 
	 */
	public static float max(float[] a) {
		check(a.length==0, EMPTY_MESSAGE);
		float amax = -Float.MAX_VALUE;
		for (float value : a)			
			if (value > amax)				
				amax = value;		
		return amax;
	}

	/** Returns the maximum value in the array
	 * 
	 * @param a - array to process
	 * @return value 
	 */
	public static double max(double[] a) {
		check(a.length==0, EMPTY_MESSAGE);
		double amax = -Double.MAX_VALUE;
		for (double value : a)			
			if (value > amax)				
				amax = value;		
		return amax;
	}

	/** Returns the minimum value in the array
	 * 
	 * @param a - array to process
	 * @return value 
	 */
	public static byte min(byte[] a) {
		check(a.length==0, EMPTY_MESSAGE);
		byte amin = Byte.MAX_VALUE;
		for (byte value : a)
			if (value < amin)
				amin = value;
		return amin;
	}

	/** Returns the minimum value in the array
	 * 
	 * @param a - array to process
	 * @return value 
	 */
	public static short min(short[] a) {
		check(a.length==0, EMPTY_MESSAGE);
		short amin = Short.MAX_VALUE;
		for (short value : a)
			if (value < amin)
				amin = value;
		return amin;
	}

	/** Returns the minimum value in the array
	 * 
	 * @param a - array to process
	 * @return value 
	 */
	public static int min(int[] a) {
		check(a.length==0, EMPTY_MESSAGE);
		int amin = Integer.MAX_VALUE;
		for (int value : a)
			if (value < amin)
				amin = value;
		return amin;
	}

	/** Returns the minimum value in the array
	 * 
	 * @param a - array to process
	 * @return value 
	 */
	public static long min(long[] a) {
		check(a.length==0, EMPTY_MESSAGE);
		long amin = Long.MAX_VALUE;
		for (long value : a)
			if (value < amin)
				amin = value;
		return amin;
	}

	/** Returns the minimum value in the array
	 * 
	 * @param a - array to process
	 * @return value 
	 */
	public static float min(float[] a) {
		check(a.length==0, EMPTY_MESSAGE);
		float amin = Float.MAX_VALUE;
		for (float value : a)
			if (value < amin)
				amin = value;
		return amin;
	}

	/** Returns the minimum value in the array
	 * 
	 * @param a - array to process
	 * @return value 
	 */
	public static double min(double[] a) {
		check(a.length==0, EMPTY_MESSAGE);
		double amin = Double.MAX_VALUE;
		for (double value : a)
			if (value < amin)
				amin = value;
		return amin;
	}
	
	/** Returns the minimum and maximum values, respectively, in the array
	 * 
	 * @param a - array to process
	 * @return value 
	 */
	public static byte[] minMax(byte[] a) {
		check(a.length==0, EMPTY_MESSAGE);
		byte amax = Byte.MIN_VALUE;
		byte amin = Byte.MAX_VALUE;		
		for (byte value : a) {
			if (value > amax)
				amax = value;
			if (value < amin)
				amin = value;
		}
		return new byte[]{amin, amax};
	}

	/** Returns the minimum and maximum values, respectively, in the array
	 * 
	 * @param a - array to process
	 * @return value 
	 */
	public static short[] minMax(short[] a) {
		check(a.length==0, EMPTY_MESSAGE);
		short amax = Short.MIN_VALUE;
		short amin = Short.MAX_VALUE;		
		for (short value : a) { 
			if (value > amax)
				amax = value;
			if (value < amin)
				amin = value;
		}
		return new short[]{amin, amax};
	}

	/** Returns the minimum and maximum values, respectively, in the array
	 * 
	 * @param a - array to process
	 * @return value 
	 */
	public static int[] minMax(int[] a) {
		check(a.length==0, EMPTY_MESSAGE);
		int amax = Integer.MIN_VALUE;
		int amin = Integer.MAX_VALUE;		
		for (int value : a) { 
			if (value > amax)
				amax = value;
			if (value < amin)
				amin = value;
		}
		return new int[]{amin, amax};
	}

	/** Returns the minimum and maximum values, respectively, in the array
	 * 
	 * @param a - array to process
	 * @return value 
	 */
	public static long[] minMax(long[] a) {
		check(a.length==0, EMPTY_MESSAGE);
		long amax = Long.MIN_VALUE;
		long amin = Long.MAX_VALUE;		
		for (long value : a) { 
			if (value > amax)
				amax = value;
			if (value < amin)
				amin = value;
		}
		return new long[]{amin, amax};
	}

	/** Returns the minimum and maximum values, respectively, in the array
	 * 
	 * @param a - array to process
	 * @return value 
	 */
	public static float[] minMax(float[] a) {
		check(a.length==0, EMPTY_MESSAGE);
		float amax = -Float.MAX_VALUE;
		float amin = Float.MAX_VALUE;
		for (float value : a) { 
			if (value > amax)
				amax = value;
			if (value < amin)
				amin = value;
		}
		return new float[]{amin, amax};
	}

	/** Returns the minimum and maximum values, respectively, in the array
	 * 
	 * @param a - array to process
	 * @return value 
	 */
	public static double[] minMax(double[] a) {
		check(a.length==0, EMPTY_MESSAGE);
		double amax = -Double.MAX_VALUE;
		double amin = Double.MAX_VALUE;
		for (double value : a) {
			if (value > amax)
				amax = value;
			if (value < amin){
				amin = value;
			}
		}
		return new double[]{amin, amax};
	}

	/** Returns the standard deviation of the array
	 * 
	 * @param a - array to process
	 * @return value 
	 */
	public static double stdev(byte[] a) {
		if (a.length==1) return 0.0;
		double av = ave(a);
		double s = 0.0;
		for (double value : a)
			s += (value-av)*(value-av);
		return Math.sqrt(s/((double)a.length-1.0));
	}

	/** Returns the standard deviation of the array
	 * 
	 * @param a - array to process
	 * @return value 
	 */
	public static double stdev(short[] a) {
		if (a.length==1) return 0.0;
		double av = ave(a);
		double s = 0.0;
		for (double value : a)
			s += (value-av)*(value-av);
		return Math.sqrt(s/((double)a.length-1.0));
	}

	/** Returns the standard deviation of the array
	 * 
	 * @param a - array to process
	 * @return value 
	 */
	public static double stdev(int[] a) {
		if (a.length==1) return 0.0;
		double av = ave(a);
		double s = 0.0;
		for (double value : a)
			s += (value-av)*(value-av);
		return Math.sqrt(s/((double)a.length-1.0));
	}

	/** Returns the standard deviation of the array
	 * 
	 * @param a - array to process
	 * @return value 
	 */
	public static double stdev(long[] a) {
		if (a.length==1) return 0.0;
		double av = ave(a);
		double s = 0.0;
		for (double value : a)
			s += (value-av)*(value-av);
		return Math.sqrt(s/((double)a.length-1.0));
	}

	/** Returns the standard deviation of the array
	 * 
	 * @param a - array to process
	 * @return value 
	 */
	public static double stdev(float[] a) {
		if (a.length==1) return 0.0;
		double av = ave(a);
		double s = 0.0;
		for (double value : a)
			s += (value-av)*(value-av);
		return Math.sqrt(s/((double)a.length-1.0));
	}
	
	/** Returns the standard deviation of the array
	 * 
	 * @param a - array to process
	 * @return value 
	 */
	public static double stdev(double[] a) {
		if (a.length==1) return 0.0;
		double av = ave(a);
		double s = 0.0;
		for (double value : a)
			s += (value-av)*(value-av);
		return Math.sqrt(s/((double)a.length-1.0));
	}

	/** Performs a linear re-scaling of the array {@code a} by converting
	 *  the values in {@code a} to go from a range of (originalMin, originalMax) 
	 *  to a range of (newMin, newMax).<p>
	 *  The equation is:<p> 
	 *  rescaled[i] = (a[i] - originalMin) * (newMax - newMin)/(originalMax - originalMin) + newMin
	 * @param a - array to process
	 * @param newMin - the minimum value that the returned array will have
	 * @param newMax - the maximum value that the returned array will have
	 * @param <T> - data type
	 * @return A new array that is re-scaled to be in the specified range
	 */
	@SuppressWarnings("unchecked")
	public static <T> T linearRescale(double[] a, double newMin, double newMax) { //, Class<T> c) {		
		//T[] anew = (T[]) Array.newInstance(c, a.length);
		double[] anew = new double[a.length];
		double[] aMinMax = minMax(a);
		double ratio = (newMax - newMin)/(aMinMax[1] - aMinMax[0]);
		for (int i=0; i<a.length; i++)			
			//anew[i] = c.cast((a[i] - aMinMax[0]) * ratio + newMin);
			anew[i] = ((a[i] - aMinMax[0]) * ratio + newMin);
		return (T)anew;
	}
	
	
	/*
	 * 
	 *  The following methods are implemented directly from java.util.Arrays
	 *  
	 */
	
	/**Returns a fixed-size list backed by the specified array.
	 * 
	 * @param a - array to process
	 * @param <T> data type
	 * @return List
	 */
	@SafeVarargs
	public static <T> List<T> asList(T... a) {return Arrays.asList(a);}

	/**Searches the specified array of bytes for the specified value using the binary search algorithm.
	 * @see java.util.Arrays#binarySearch(byte[] a, byte key) */
	public static int binarySearch(byte[] a, byte key) {return Arrays.binarySearch(a, key);}

	/**Searches a range of the specified array of bytes for the specified value using the binary search algorithm.
	 * @see java.util.Arrays#binarySearch(byte[] a, int fromIndex, int toIndex, byte key)} */
	public static int binarySearch(byte[] a, int fromIndex, int toIndex, byte key) {return Arrays.binarySearch(a, fromIndex, toIndex, key);}

	/**Searches the specified array of chars for the specified value using the binary search algorithm.
	 * @see java.util.Arrays#binarySearch(char[] a, char key)} */
	public static int binarySearch(char[] a, char key) {return Arrays.binarySearch(a, key);}

	/**Searches a range of the specified array of chars for the specified value using the binary search algorithm. 
	 * @see java.util.Arrays#binarySearch(char[] a, int fromIndex, int toIndex, char key)} */
	public static int binarySearch(char[] a, int fromIndex, int toIndex, char key) {return Arrays.binarySearch(a, fromIndex, toIndex, key);}

	/**Searches the specified array of doubles for the specified value using the binary search algorithm.
	 * @see java.util.Arrays#binarySearch(double[] a, double key)} */
	public static int binarySearch(double[] a, double key) {return Arrays.binarySearch(a, key);}

	/**Searches a range of the specified array of doubles for the specified value using the binary search algorithm.
	 * @see java.util.Arrays#binarySearch(double[] a, int fromIndex, int toIndex, double key)} */
	public static int binarySearch(double[] a, int fromIndex, int toIndex, double key) {return Arrays.binarySearch(a, fromIndex, toIndex, key);}

	/**Searches the specified array of floats for the specified value using the binary search algorithm.
	 * @see java.util.Arrays#binarySearch(float[] a, float key)} */
	public static int binarySearch(float[] a, float key) {return Arrays.binarySearch(a, key);}

	/**Searches a range of the specified array of floats for the specified value using the binary search algorithm.
	 * @see java.util.Arrays#binarySearch(float[] a, int fromIndex, int toIndex, float key) */
	public static int binarySearch(float[] a, int fromIndex, int toIndex, float key) {return Arrays.binarySearch(a, fromIndex, toIndex, key);}

	/**Searches the specified array of ints for the specified value using the binary search algorithm.
	 * @see java.util.Arrays#binarySearch(int[] a, int key)} */
	public static int binarySearch(int[] a, int key) {return Arrays.binarySearch(a, key);}

	/**Searches a range of the specified array of ints for the specified value using the binary search algorithm.
	 * @see java.util.Arrays#binarySearch(int[] a, int fromIndex, int toIndex, int key)} */
	public static int binarySearch(int[] a, int fromIndex, int toIndex, int key) {return Arrays.binarySearch(a, fromIndex, toIndex, key);}

	/**Searches a range of the specified array of longs for the specified value using the binary search algorithm.
	 * @see java.util.Arrays#binarySearch(long[] a, int fromIndex, int toIndex, long key)} */
	public static int binarySearch(long[] a, int fromIndex, int toIndex, long key) {return Arrays.binarySearch(a, fromIndex, toIndex, key);}

	/**Searches the specified array of longs for the specified value using the binary search algorithm.
	 * @see java.util.Arrays#binarySearch(long[] a, long key)} */
	public static int binarySearch(long[] a, long key) {return Arrays.binarySearch(a, key);}

	/**Searches a range of the specified array for the specified object using the binary search algorithm.
	 * @see java.util.Arrays#binarySearch(Object[] a, int fromIndex, int toIndex, Object key)} */
	public static int binarySearch(Object[] a, int fromIndex, int toIndex, Object key) {return Arrays.binarySearch(a, fromIndex, toIndex, key);}

	/**Searches the specified array for the specified object using the binary search algorithm.
	 * @see java.util.Arrays#binarySearch(Object[] a, Object key)} */
	public static int binarySearch(Object[] a, Object key) {return Arrays.binarySearch(a, key);}

	/**Searches a range of the specified array of shorts for the specified value using the binary search algorithm.
	 * @see java.util.Arrays#binarySearch(short[] a, int fromIndex, int toIndex, short key)} */
	public static int binarySearch(short[] a, int fromIndex, int toIndex, short key) {return Arrays.binarySearch(a, fromIndex, toIndex, key);}

	/**Searches the specified array of shorts for the specified value using the binary search algorithm.
	 * @see java.util.Arrays#binarySearch(short[] a, short key)} */
	public static int binarySearch(short[] a, short key) {return Arrays.binarySearch(a, key);}

	/**Searches a range of the specified array for the specified object using the binary search algorithm.
	 * @see java.util.Arrays#binarySearch(T[] a, int fromIndex, int toIndex, T key, Comparator c)} */
	public static <T> int binarySearch(T[] a, int fromIndex, int toIndex, T key, Comparator<? super T> c) {return Arrays.binarySearch(a, fromIndex, toIndex, key, c);}

	/**Searches the specified array for the specified object using the binary search algorithm.
	 * @see java.util.Arrays#binarySearch(T[] a, T key, Comparator c) */ 
	public static <T> int binarySearch(T[] a, T key, Comparator<? super T> c) {return Arrays.binarySearch(a, key, c);}

	/**Copies the specified array, truncating or padding with false (if necessary) so the copy has the specified length.
	 * @see java.util.Arrays#copyOf(boolean[] original, int newLength) */ 
	public static boolean[] copyOf(boolean[] original, int newLength) {return Arrays.copyOf(original, newLength);}

	/**Copies the specified array, truncating or padding with zeros (if necessary) so the copy has the specified length.
	 * @see java.util.Arrays#copyOf(byte[] original, int newLength) */ 
	public static byte[] copyOf(byte[] original, int newLength) {return Arrays.copyOf(original, newLength);}

	/**Copies the specified array, truncating or padding with null characters (if necessary) so the copy has the specified length.
	 * @see java.util.Arrays#copyOf(char[] original, int newLength) */ 
	public static char[] copyOf(char[] original, int newLength) {return Arrays.copyOf(original, newLength);}

	/**Copies the specified array, truncating or padding with zeros (if necessary) so the copy has the specified length.
	 * @see java.util.Arrays#copyOf(double[] original, int newLength) */ 
	public static double[] copyOf(double[] original, int newLength) {return Arrays.copyOf(original, newLength);}

	/**Copies the specified array, truncating or padding with zeros (if necessary) so the copy has the specified length.
	 * @see java.util.Arrays#copyOf(float[] original, int newLength) */ 
	public static float[] copyOf(float[] original, int newLength) {return Arrays.copyOf(original, newLength);}

	/**Copies the specified array, truncating or padding with zeros (if necessary) so the copy has the specified length.
	 * @see java.util.Arrays#copyOf(int[] original, int newLength) */ 
	public static int[] copyOf(int[] original, int newLength) {return Arrays.copyOf(original, newLength);}

	/**Copies the specified array, truncating or padding with zeros (if necessary) so the copy has the specified length.
	 * @see java.util.Arrays#copyOf(long[] original, int newLength) */ 
	public static long[] copyOf(long[] original, int newLength) {return Arrays.copyOf(original, newLength);}

	/**Copies the specified array, truncating or padding with zeros (if necessary) so the copy has the specified length.
	 * @see java.util.Arrays#copyOf(short[] original, int newLength) */ 
	public static short[] copyOf(short[] original, int newLength) {return Arrays.copyOf(original, newLength);}

	/**Copies the specified array, truncating or padding with nulls (if necessary) so the copy has the specified length.
	 * @see java.util.Arrays#copyOf(T[] original, int newLength) */ 
	public static <T> T[] copyOf(T[] original, int newLength) {return Arrays.copyOf(original, newLength);}

	/**Copies the specified array, truncating or padding with nulls (if necessary) so the copy has the specified length.
	 * @see java.util.Arrays#copyOf(U[] original, int newLength, Class newType) */ 
	public static <T,U> T[] copyOf(U[] original, int newLength, Class<? extends T[]> newType) {return Arrays.copyOf(original, newLength, newType);}

	/**Copies the specified range of the specified array into a new array.
	 * @see java.util.Arrays#copyOfRange(boolean[] original, int from, int to) */ 
	public static boolean[] copyOfRange(boolean[] original, int from, int to) {return Arrays.copyOfRange(original, from, to);}

	/**Copies the specified range of the specified array into a new array.
	 * @see java.util.Arrays#copyOfRange(byte[] original, int from, int to) */ 
	public static byte[] copyOfRange(byte[] original, int from, int to) {return Arrays.copyOfRange(original, from, to);}

	/**Copies the specified range of the specified array into a new array.
	 * @see java.util.Arrays#copyOfRange(char[] original, int from, int to) */ 
	public static char[] copyOfRange(char[] original, int from, int to) {return Arrays.copyOfRange(original, from, to);}

	/**Copies the specified range of the specified array into a new array.
	 * @see java.util.Arrays#copyOfRange(double[] original, int from, int to) */ 
	public static double[]	copyOfRange(double[] original, int from, int to) {return Arrays.copyOfRange(original, from, to);}

	/**Copies the specified range of the specified array into a new array.
	 * @see java.util.Arrays#copyOfRange(float[] original, int from, int to) */ 
	public static float[] copyOfRange(float[] original, int from, int to) {return Arrays.copyOfRange(original, from, to);}

	/**Copies the specified range of the specified array into a new array.
	 * @see java.util.Arrays#copyOfRange(int[] original, int from, int to) */ 
	public static int[] copyOfRange(int[] original, int from, int to) {return Arrays.copyOfRange(original, from, to);}

	/**Copies the specified range of the specified array into a new array.
	 * @see java.util.Arrays#copyOfRange(long[] original, int from, int to) */ 
	public static long[] copyOfRange(long[] original, int from, int to) {return Arrays.copyOfRange(original, from, to);}

	/**Copies the specified range of the specified array into a new array.
	 * @see java.util.Arrays#copyOfRange(short[] original, int from, int to) */ 
	public static short[] copyOfRange(short[] original, int from, int to) {return Arrays.copyOfRange(original, from, to);}

	/**Copies the specified range of the specified array into a new array.
	 * @see java.util.Arrays#copyOfRange(T[] original, int from, int to) */ 
	public static <T> T[] copyOfRange(T[] original, int from, int to) {return Arrays.copyOfRange(original, from, to);}

	/**Copies the specified range of the specified array into a new array.
	 * @see java.util.Arrays#copyOfRange(U[] original, int from, int to, Class newType) */ 
	public static <T,U> T[] copyOfRange(U[] original, int from, int to, Class<? extends T[]> newType) {return Arrays.copyOfRange(original, from, to, newType);}

	/**Returns true if the two specified arrays are deeply equal to one another.
	 * @see java.util.Arrays#deepEquals(Object[] a1, Object[] a2) */ 
	public static boolean deepEquals(Object[] a1, Object[] a2) {return Arrays.deepEquals(a1, a2);}

	/**Returns a hash code based on the "deep contents" of the specified array.
	 * @see java.util.Arrays#deepHashCode(Object[] a) */ 
	public static int deepHashCode(Object[] a) {return Arrays.deepHashCode(a);}

	/**Returns a string representation of the "deep contents" of the specified array.
	 * @see java.util.Arrays#deepToString(Object[] a) */ 
	public static String deepToString(Object[] a) {return Arrays.deepToString(a);}

	/**Returns true if the two specified arrays of booleans are equal to one another.
	 * @see java.util.Arrays#equals(boolean[] a, boolean[] a2) */ 
	public static boolean equals(boolean[] a, boolean[] a2) {return Arrays.equals(a, a2);}

	/**Returns true if the two specified arrays of bytes are equal to one another.
	 * @see java.util.Arrays#equals(byte[] a, byte[] a2) */ 
	public static boolean equals(byte[] a, byte[] a2) {return Arrays.equals(a, a2);}

	/**Returns true if the two specified arrays of chars are equal to one another.
	 * @see java.util.Arrays#equals(char[] a, char[] a2) */ 
	public static boolean equals(char[] a, char[] a2) {return Arrays.equals(a, a2);}

	/**Returns true if the two specified arrays of doubles are equal to one another.
	 * @see java.util.Arrays#equals(double[] a, double[] a2) */ 
	public static boolean equals(double[] a, double[] a2) {return Arrays.equals(a, a2);}

	/**Returns true if the two specified arrays of floats are equal to one another.
	 * @see java.util.Arrays#equals(float[] a, float[] a2) */ 
	public static boolean equals(float[] a, float[] a2) {return Arrays.equals(a, a2);}

	/**Returns true if the two specified arrays of ints are equal to one another.
	 * @see java.util.Arrays#equals(int[] a, int[] a2) */ 
	public static boolean equals(int[] a, int[] a2) {return Arrays.equals(a, a2);}

	/**Returns true if the two specified arrays of longs are equal to one another.
	 * @see java.util.Arrays#equals(long[] a, long[] a2) */ 
	public static boolean equals(long[] a, long[] a2) {return Arrays.equals(a, a2);}

	/**Returns true if the two specified arrays of Objects are equal to one another.
	 * @see java.util.Arrays#equals(Object[] a, Object[] a2) */ 
	public static boolean equals(Object[] a, Object[] a2) {return Arrays.equals(a, a2);}

	/**Returns true if the two specified arrays of shorts are equal to one another.
	 * @see java.util.Arrays#equals(short[] a, short[] a2) */ 
	public static boolean equals(short[] a, short[] a2) {return Arrays.equals(a, a2);}

	/**Assigns the specified boolean value to each element of the specified array of booleans.
	 * @see java.util.Arrays#fill(boolean[] a, boolean val) */ 
	public static void	fill(boolean[] a, boolean val) {Arrays.fill(a, val);}

	/**Assigns the specified boolean value to each element of the specified range of the specified array of booleans.
	 * @see java.util.Arrays#fill(boolean[] a, int fromIndex, int toIndex, boolean val) */ 
	public static void fill(boolean[] a, int fromIndex, int toIndex, boolean val) {Arrays.fill(a, fromIndex, toIndex, val);}

	/**Assigns the specified byte value to each element of the specified array of bytes.
	 * @see java.util.Arrays#fill(byte[] a, byte val) */ 
	public static void fill(byte[] a, byte val) {Arrays.fill(a, val);}

	/**Assigns the specified byte value to each element of the specified range of the specified array of bytes.
	 * @see java.util.Arrays#fill(byte[] a, int fromIndex, int toIndex, byte val) */ 
	public static void fill(byte[] a, int fromIndex, int toIndex, byte val) {Arrays.fill(a, fromIndex, toIndex, val);}

	/**Assigns the specified char value to each element of the specified array of chars.
	 * @see java.util.Arrays#fill(char[] a, char val) */ 
	public static void fill(char[] a, char val) {Arrays.fill(a, val);}

	/**Assigns the specified char value to each element of the specified range of the specified array of chars.
	 * @see java.util.Arrays#fill(char[] a, int fromIndex, int toIndex, char val) */ 
	public static void fill(char[] a, int fromIndex, int toIndex, char val) {Arrays.fill(a, fromIndex, toIndex, val);}

	/**Assigns the specified double value to each element of the specified array of doubles.
	 * @see java.util.Arrays#fill(double[] a, double val) */ 
	public static void fill(double[] a, double val) {Arrays.fill(a, val);}

	/**Assigns the specified double value to each element of the specified range of the specified array of doubles.
	 * @see java.util.Arrays#fill(double[] a, int fromIndex, int toIndex, double val) */ 
	public static void fill(double[] a, int fromIndex, int toIndex, double val) {Arrays.fill(a, fromIndex, toIndex, val);}

	/**Assigns the specified float value to each element of the specified array of floats.
	 * @see java.util.Arrays#fill(float[] a, float val) */ 
	public static void fill(float[] a, float val) {Arrays.fill(a, val);}

	/**Assigns the specified float value to each element of the specified range of the specified array of floats.
	 * @see java.util.Arrays#fill(float[] a, int fromIndex, int toIndex, float val) */ 
	public static void fill(float[] a, int fromIndex, int toIndex, float val) {Arrays.fill(a, fromIndex, toIndex, val);}

	/**Assigns the specified int value to each element of the specified array of ints.
	 * @see java.util.Arrays#fill(int[] a, int val) */ 
	public static void fill(int[] a, int val) {Arrays.fill(a, val);}

	/**Assigns the specified int value to each element of the specified range of the specified array of ints.
	 * @see java.util.Arrays#fill(int[] a, int fromIndex, int toIndex, int val) */ 
	public static void fill(int[] a, int fromIndex, int toIndex, int val) {Arrays.fill(a, fromIndex, toIndex, val);}

	/**Assigns the specified long value to each element of the specified range of the specified array of longs.
	 * @see java.util.Arrays#fill(long[] a, int fromIndex, int toIndex, long val) */ 
	public static void fill(long[] a, int fromIndex, int toIndex, long val) {Arrays.fill(a, fromIndex, toIndex, val);}

	/**Assigns the specified long value to each element of the specified array of longs.
	 * @see java.util.Arrays#fill(long[] a, long val) */ 
	public static void fill(long[] a, long val) {Arrays.fill(a, val);}

	/**Assigns the specified Object reference to each element of the specified range of the specified array of Objects.
	 * @see java.util.Arrays#fill(Object[] a, int fromIndex, int toIndex, Object val) */ 
	public static void fill(Object[] a, int fromIndex, int toIndex, Object val) {Arrays.fill(a, fromIndex, toIndex, val);}

	/**Assigns the specified Object reference to each element of the specified array of Objects.
	 * @see java.util.Arrays#fill(Object[] a, Object val) */ 
	public static void fill(Object[] a, Object val) {Arrays.fill(a, val);}

	/**Assigns the specified short value to each element of the specified range of the specified array of shorts.
	 * @see java.util.Arrays#fill(short[] a, int fromIndex, int toIndex, short val) */ 
	public static void fill(short[] a, int fromIndex, int toIndex, short val) {Arrays.fill(a, fromIndex, toIndex, val);}

	/**Assigns the specified short value to each element of the specified array of shorts.
	 * @see java.util.Arrays#fill(short[] a, short val) */ 
	public static void fill(short[] a, short val) {Arrays.fill(a, val);}

	/**Returns a hash code based on the contents of the specified array.
	 * @see java.util.Arrays#hashCode(boolean[] a) */ 
	public static int hashCode(boolean[] a) {return Arrays.hashCode(a);}

	/**Returns a hash code based on the contents of the specified array.
	 * @see java.util.Arrays#hashCode(byte[] a) */ 
	public static int hashCode(byte[] a) {return Arrays.hashCode(a);}

	/**Returns a hash code based on the contents of the specified array.
	 * @see java.util.Arrays#hashCode(char[] a) */ 
	public static int hashCode(char[] a) {return Arrays.hashCode(a);}

	/**Returns a hash code based on the contents of the specified array.
	 * @see java.util.Arrays#hashCode(double[] a) */ 
	public static int hashCode(double[] a) {return Arrays.hashCode(a);}

	/**Returns a hash code based on the contents of the specified array.
	 * @see java.util.Arrays#hashCode(float[] a) */ 
	public static int hashCode(float[] a) {return Arrays.hashCode(a);}

	/**Returns a hash code based on the contents of the specified array.
	 * @see java.util.Arrays#hashCode(int[] a) */ 
	public static int hashCode(int[] a) {return Arrays.hashCode(a);}

	/**Returns a hash code based on the contents of the specified array.
	 * @see java.util.Arrays#hashCode(long[] a) */ 
	public static int hashCode(long[] a) {return Arrays.hashCode(a);}

	/**Returns a hash code based on the contents of the specified array.
	 * @see java.util.Arrays#hashCode(Object[] a) */ 
	public static int hashCode(Object[] a) {return Arrays.hashCode(a);}

	/**Returns a hash code based on the contents of the specified array.
	 * @see java.util.Arrays#hashCode(short[] a) */ 
	public static int hashCode(short[] a) {return Arrays.hashCode(a);}

	/**Sorts the specified array into ascending numerical order.
	 * @see java.util.Arrays#sort(byte[] a) */ 
	public static void sort(byte[] a) {Arrays.sort(a);}

	/**Sorts the specified range of the array into ascending order.
	 * @see java.util.Arrays#sort(byte[] a, int fromIndex, int toIndex) */ 
	public static void sort(byte[] a, int fromIndex, int toIndex) {Arrays.sort(a, fromIndex, toIndex);}

	/**Sorts the specified array into ascending numerical order.
	 * @see java.util.Arrays#sort(char[] a) */ 
	public static void sort(char[] a) {Arrays.sort(a);}

	/**Sorts the specified range of the array into ascending order.
	 * @see java.util.Arrays#sort(char[] a, int fromIndex, int toIndex) */ 
	public static void sort(char[] a, int fromIndex, int toIndex) {Arrays.sort(a, fromIndex, toIndex);}

	/**Sorts the specified array into ascending numerical order.
	 * @see java.util.Arrays#sort(double[] a) */ 
	public static void sort(double[] a) {Arrays.sort(a);}

	/**Sorts the specified range of the array into ascending order.
	 * @see java.util.Arrays#sort(double[] a, int fromIndex, int toIndex) */ 
	public static void sort(double[] a, int fromIndex, int toIndex) {Arrays.sort(a, fromIndex, toIndex);}

	/**Sorts the specified array into ascending numerical order.
	 * @see java.util.Arrays#sort(float[] a) */ 
	public static void sort(float[] a) {Arrays.sort(a);}

	/**Sorts the specified range of the array into ascending order.
	 * @see java.util.Arrays#sort(float[] a, int fromIndex, int toIndex) */ 
	public static void sort(float[] a, int fromIndex, int toIndex) {Arrays.sort(a, fromIndex, toIndex);}

	/**Sorts the specified array into ascending numerical order.
	 * @see java.util.Arrays#sort(int[] a) */ 
	public static void sort(int[] a) {Arrays.sort(a);}

	/**Sorts the specified range of the array into ascending order.
	 * @see java.util.Arrays#sort(int[] a, int fromIndex, int toIndex) */ 
	public static void sort(int[] a, int fromIndex, int toIndex) {Arrays.sort(a, fromIndex, toIndex);}

	/**Sorts the specified array into ascending numerical order.
	 * @see java.util.Arrays#sort(long[] a) */ 
	public static void sort(long[] a) {Arrays.sort(a);}

	/**Sorts the specified range of the array into ascending order.
	 * @see java.util.Arrays#sort(long[] a, int fromIndex, int toIndex) */ 
	public static void sort(long[] a, int fromIndex, int toIndex) {Arrays.sort(a, fromIndex, toIndex);}

	/**Sorts the specified array of objects into ascending order, according to the natural ordering of its elements.
	 * @see java.util.Arrays#sort(Object[] a) */ 
	public static void sort(Object[] a) {Arrays.sort(a);}

	/**Sorts the specified range of the specified array of objects into ascending order, according to the natural ordering of its elements.
	 * @see java.util.Arrays#sort(Object[] a, int fromIndex, int toIndex) */ 
	public static void sort(Object[] a, int fromIndex, int toIndex) {Arrays.sort(a, fromIndex, toIndex);}

	/**Sorts the specified array into ascending numerical order.
	 * @see java.util.Arrays#sort(short[] a) */ 
	public static void sort(short[] a) {Arrays.sort(a);}

	/**Sorts the specified range of the array into ascending order.
	 * @see java.util.Arrays#sort(short[] a, int fromIndex, int toIndex) */ 
	public static void sort(short[] a, int fromIndex, int toIndex) {Arrays.sort(a, fromIndex, toIndex);}

	/**Sorts the specified array of objects according to the order induced by the specified comparator.
	 * @see java.util.Arrays#sort(T[] a, Comparator c) */ 
	public static <T> void sort(T[] a, Comparator<? super T> c) {Arrays.sort(a, c);}

	/**Sorts the specified range of the specified array of objects according to the order induced by the specified comparator.
	 * @see java.util.Arrays#sort(T[] a, int fromIndex, int toIndex, Comparator c) */ 
	public static <T> void sort(T[] a, int fromIndex, int toIndex, Comparator<? super T> c) {Arrays.sort(a, fromIndex, toIndex, c);}

	/**Returns a string representation of the contents of the specified array.
	 * @see java.util.Arrays#toString(boolean[] a) */ 
	public static String toString(boolean[] a) {return Arrays.toString(a);}

	/**Returns a string representation of the contents of the specified array.
	 * @see java.util.Arrays#toString(byte[] a) */ 
	public static String toString(byte[] a) {return Arrays.toString(a);}

	/**Returns a string representation of the contents of the specified array.
	 * @see java.util.Arrays#toString(char[] a) */ 
	public static String toString(char[] a) {return Arrays.toString(a);}

	/**Returns a string representation of the contents of the specified array.
	 * @see java.util.Arrays#toString(double[] a) */ 
	public static String toString(double[] a) {return Arrays.toString(a);}

	/**Returns a string representation of the contents of the specified array.
	 * @see java.util.Arrays#toString(float[] a) */ 
	public static String toString(float[] a) {return Arrays.toString(a);}

	/**Returns a string representation of the contents of the specified array.
	 * @see java.util.Arrays#toString(int[] a) */ 
	public static String toString(int[] a) {return Arrays.toString(a);}

	/**Returns a string representation of the contents of the specified array.
	 * @see java.util.Arrays#toString(long[] a) */ 
	public static String toString(long[] a) {return Arrays.toString(a);}

	/**Returns a string representation of the contents of the specified array.
	 * @see java.util.Arrays#toString(Object[] a) */ 
	public static String toString(Object[] a) {return Arrays.toString(a);}

	/**Returns a string representation of the contents of the specified array.
	 * @see java.util.Arrays#toString(short[] a) */ 
	public static String toString(short[] a) {return Arrays.toString(a);}

}