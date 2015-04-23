package org.lemming.tests;

import java.io.File;
import javolution.util.FastMap;

import org.junit.Before;
import org.junit.Test;
import org.lemming.data.ExtendableTable;
import org.lemming.outputs.GenericPrintToFile;

@SuppressWarnings("javadoc")
public class ExtendableTableTest {

	@Before
	public void setUp() throws Exception {
	}

	@Test
	public void testExtendableTable() {
		int N = (int) 1e7;
		ExtendableTable h = new ExtendableTable();
		h.addNewMember("x");
		h.addNewMember("y");
		h.addNewMember("z");
		h.addNewMember("frame");
		h.addNewMember("roi");
		
		long t0 = System.currentTimeMillis();
		long t00 = t0;
		long[] T = new long[N];
		
		long sum = 0;
		long max = 0;
		long id = 0;			
		
		
		for (int i=0;i<N;i++){
			FastMap<String,Object> gi = h.newRow();
			gi.put("x", i);
			gi.put("y", -i);
			gi.put("z", i);
			gi.put("frame", i+1);
			gi.put("roi", new double[] {1,2,3,4});
			h.addRow(gi);
			
			long ct = System.currentTimeMillis();
			long dt = ct - t0;
			
			t0=ct;
			
			sum+=dt;
			
			if (max < dt) {
				max=dt;
				id=i;
			}
			
			T[i] = dt;
		}
		
		System.out.println("Total time is "+(System.currentTimeMillis()-t00));
		System.out.println("loop is "+ N);
		System.out.println("Average time is "+sum/N);
		System.out.println("Max time is "+max);		
		System.out.println("row is "+id+":"+T[(int) id]);	
		System.gc();
	}
	
	@Test
	public void stressTestFIFO() {
		ExtendableTable h = new ExtendableTable();
		
		h.addNewMember("x");
		h.addNewMember("y");
		h.addNewMember("z");
		h.addNewMember("id"); // !!
		h.addNewMember("frame");
		h.addNewMember("roi");
		
		int N = (int) 1e7;
		long t0 = System.currentTimeMillis();
		
		for (int i=0; i<N; i++) {
			FastMap<String,Object> gi = h.newRow();
			gi.put("x", i);
			gi.put("y", -i);
			gi.put("z", i);
			gi.put("id", i);
			gi.put("frame", i+1);
			gi.put("roi", new double[] {1,2,3,4});
			h.addRow(gi);
		}
		
		
		System.out.println("Elapsed time "+ (System.currentTimeMillis()-t0));
		System.out.println("Rows:"+h.getNumberOfRows());
		/*PrintToScreen ps = new PrintToScreen();
		ps.setInput(h.getFIFO());
		
		Thread t_print = new Thread(ps,"PrintToScreen");*/
		
		GenericPrintToFile pf = new GenericPrintToFile(new File("/tmp/stressTestFIFO2.csv"));
		pf.setInput(h.getFIFO());
		
		Thread t_file = new Thread(pf,"PrintToFile");
		
		//t_print.start();
		t_file.start();
		try {
			//t_print.join();
			t_file.join();
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
	}


}
