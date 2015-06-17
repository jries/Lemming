package org.lemming.pipeline;

import ij.util.ArrayUtil;

import java.util.Collections;
import java.util.List;

import org.junit.Before;
import org.junit.Test;

public class ExtendableTableTest {

	@Before
	public void setUp() throws Exception {
	}

	@Test
	public void test() {
		int N = (int) 1e7;
		ExtendableTable h = new ExtendableTable();
		h.addNewMember("z");
		h.addNewMember("frame");
		h.addNewMember("roi");
		
		long t0 = System.nanoTime();
		long t00 = t0;
		final long[] T = new long[N];
		
		long sum = 0;
		long max = 0;
		long id = 0;			
		
		List<Object> colx = h.getColumn("xpix");
		List<Object> coly = h.getColumn("ypix");
		List<Object> colz = h.getColumn("z");
		List<Object> colf = h.getColumn("frame");
		List<Object> colr = h.getColumn("roi");
		double[] D = new double[] {1,2,3,4};
		
		
		for (Integer i=0;i<N;i++){
			colx.add(i);
			coly.add(-i);
			colz.add(i);
			colf.add(i+1);
			colr.add(D);			
			
			long ct = System.nanoTime();
			long dt = ct - t0;
			
			t0=ct;
			
			sum+=dt;
			
			if (max < dt) {
				max=dt;
				id=i;
			}
			
			T[i] = dt;
		}
		
		System.out.println("Total time is "+(System.nanoTime()-t00)/1e6);
		System.out.println("Average time is "+sum/N/1e6);
		System.out.println("Max time is "+max/1e6);		
		System.out.println("row is "+id);	
		System.gc();
	}

}
