package org.lemming.math;

import java.util.Collections;
import java.util.List;
import java.util.Random;

import net.imglib2.type.numeric.RealType;

/**
 * Median algorithms QuickSort and QuickSelect
 * @author ronny
 *
 */
public class QuickSelect {
	public static <T extends RealType<T>> Long binapprox(List<T> values) {
		double sum = 0d;
		for (T i : values)
			sum += i.getRealDouble();
		double mean = sum / values.size();
		double sumDev = 0;
		for (T i : values)
			sumDev += Math.pow(i.getRealDouble() - mean, 2);
		double stdDev = Math.sqrt(sumDev / values.size());

		return Math.round(stdDev);
	}

	public static <T extends Comparable<T>> T select(final List<T> values,
			final int kin) {
		int k = kin;
		int left = 0;
		int right = values.size() - 1;
		Random rand = new Random();
		while (right >= left) {
			int partionIndex = rand.nextInt(right - left + 1) + left;
			int newIndex = partition(values, left, right, partionIndex);
			int q = newIndex - left + 1;
			if (k == q) {
				return values.get(newIndex);
			} else if (k < q) {
				right = newIndex - 1;
			} else {
				k -= q;
				left = newIndex + 1;
			}
		}
		return null;
	}

	private static <T extends Comparable<T>> int partition(
			final List<T> values, final int left, final int right,
			final int partitionIndex) {
		T partionValue = values.get(partitionIndex);
		int newIndex = left;
		T temp = values.get(partitionIndex);
		values.set(partitionIndex, values.get(right));
		values.set(right, temp);
		for (int i = left; i < right; i++) {
			if (values.get(i).compareTo(partionValue) < 0) {
				temp = values.get(i);
				values.set(i, values.get(newIndex));
				values.set(newIndex, temp);
				newIndex++;
			}
		}
		temp = values.get(right);
		values.set(right, values.get(newIndex));
		values.set(newIndex, temp);
		return newIndex;
	}

	/**
	 * Quicksort with median-of-three partitioning functions nearly the same as
	 * normal quicksort with the only difference being how the pivot item is
	 * selected. In normal quicksort the first element is automatically the
	 * pivot item. This causes normal quicksort to function very inefficiently
	 * when presented with an already sorted list. The division will always end
	 * up producing one sub-array with no elements and one with all the elements
	 * (minus of course the pivot item). In quicksort with median-of-three
	 * partitioning the pivot item is selected as the median between the first
	 * element, the last element, and the middle element (decided using integer
	 * division of n/2). In the cases of already sorted lists this should take
	 * the middle element as the pivot thereby reducing the inefficiency found in
	 * normal quicksort.
	 * Places the kth smallest item in a[k-1].
	 * 
	 * @param a
	 *            an array of Comparable items.
	 * @param k
	 *            the desired rank (1 is minimum) in the entire array.
	 * @param T	
	 * 			  data type
	 * @return
	 */
	public static <T extends Comparable<T>> T three(final List<T> a, final int k) {
		three(a, 0, a.size() - 1, k);
		return a.get(k);
	}

	/**
	 * Internal selection method that makes recursive calls. Uses
	 * median-of-three partitioning and a cutoff of 10. Places the kth smallest
	 * item in a[k-1].
	 * 
	 * @param a
	 *            an array of Comparable items.
	 * @param low
	 *            the left-most index of the subarray.
	 * @param high
	 *            the right-most index of the subarray.
	 * @param k
	 *            the desired rank (1 is minimum) in the entire array.
	 * @return
	 */
	private static <T extends Comparable<T>> void three(List<T> a,
			final int low, final int high, final int k) {
		if (low + CUTOFF > high)
			insertionSort(a, low, high);
		else {
			// Sort low, middle, high
			int middle = (low + high) / 2;
			if (a.get(middle).compareTo(a.get(low)) < 0)
				Collections.swap(a, low, middle);
			if (a.get(high).compareTo(a.get(low)) < 0)
				Collections.swap(a, low, high);
			if (a.get(high).compareTo(a.get(middle)) < 0)
				Collections.swap(a, middle, high);

			// Place pivot at position high - 1
			Collections.swap(a, middle, high - 1);
			T pivot = a.get(high - 1);

			// Begin partitioning
			int i, j;
			for (i = low, j = high - 1;;) {
				while (a.get(++i).compareTo(pivot) < 0)
					;
				while (pivot.compareTo(a.get(--j)) < 0)
					;
				if (i >= j)
					break;
				Collections.swap(a, i, j);
			}

			// Restore pivot
			Collections.swap(a, i, high - 1);

			// Recurse; only this part changes
			if (k <= i)
				three(a, low, i - 1, k);
			else if (k > i + 1)
				three(a, i + 1, high, k);
		}
	}

	/**
	 * Internal insertion sort routine for subarrays that is used by quicksort.
	 * 
	 * @param a
	 *            an array of Comparable items.
	 * @param low
	 *            the left-most index of the subarray.
	 * @param n
	 *            the number of items to sort.
	 */
	private static <T extends Comparable<T>> void insertionSort(List<T> a,
			final int low, final int high) {
		for (int p = low + 1; p <= high; p++) {
			T tmp = a.get(p);
			int j;

			for (j = p; j > low && tmp.compareTo(a.get(j - 1)) < 0; j--)
				a.set(j, a.get(j - 1));
			a.set(j, tmp);
		}
	}

	private static final int CUTOFF = 10;
	
	
	private static <T extends Comparable<T>> void adjustTriplet(List<T> a, int i, int step){
		int j = i+step;
		int k = i+2;
	
		if(a.get(i).compareTo(a.get(j)) < 0){
			if(a.get(k).compareTo(a.get(i)) < 0){
				Collections.swap(a,i,j);
			} else if (a.get(k).compareTo(a.get(j)) < 0){
				Collections.swap(a,j,k);
			}
		} else {
			if(a.get(i).compareTo(a.get(k)) < 0){
				Collections.swap(a,i,j);
			} else if(a.get(k).compareTo(a.get(j)) > 0){
				Collections.swap(a,j,k);
			}
		}
	}
	
	private static <T extends Comparable<T>> void selectionSort(List<T> a, int left, int size, int step){
		int min;
		for(int i=left;i<left+(size-1)*step;i=i+step){
			min = i;
			for(int j=i+step;j<left+size*step;j=j+step){
				if(a.get(j).compareTo(a.get(min)) < 0){
					min = j;
				}
			}
			Collections.swap(a,i,min);
		}
	}
	
	public static <T extends Comparable<T>> T fastmedian(List<T> A, int dim){
		
		///////////////////////////////////////////
		/// Size of the array
		int size = dim;
	
		///////////////////////////////////////////
		///  Median calculation
		int LeftToRight = 0;
		
		// Parameters
		int threshold = 2;                                    					  // pass as argument !!!!!!!!!
		
		// Definitions
		int left = 0;
		int rem = 0;            
		int step = 1;
		int i,j;
		T median;
		
		/// Run
		while(size > threshold){
			LeftToRight = 1 - LeftToRight;
			rem = size%3;
			if(LeftToRight == 1){
				i = left;
			} else {
				i = left+(3+rem)*step;
			}
			for(j = 0; j<(size/3-1);j++){
				adjustTriplet(A,i,step);
				i = i + 3*step;
			}
			if(LeftToRight == 1){
				left = left + step;
			} else {
				i = left;
				left = left + (1+rem)*step;
			}
			selectionSort(A,i,3+rem,step);
			if(rem == 2){
				if (LeftToRight == 1){
					Collections.swap(A,i+step,i+2*step);
				} else {
					Collections.swap(A,i+2*step,i+3*step);
				}
			}
			step = 3*step;
			size = size/3;
		}
		selectionSort(A,left,size,step);
		median = A.get(left + (step*(size-1)/2));	
		
		// return median value
		return median;
	}
}
