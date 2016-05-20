package org.lemming.math;

import java.util.Collections;
import java.util.List;

/**
 * Median algorithms QuickSort and QuickSelect
 * @author ronny
 *
 */
public class QuickSelect {

	
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
		int rem;
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
