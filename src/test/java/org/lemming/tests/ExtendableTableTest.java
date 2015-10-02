package org.lemming.tests;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.List;
import java.util.Random;

import org.junit.Before;
import org.junit.Test;
import org.lemming.pipeline.ExtendableTable;

import ij.IJ;

public class ExtendableTableTest {
	
	final int N = (int) 1e6;
	final long[] T = new long[N];
	long sum = 0;
	long max = 0;
	long id = 0;
	long t0,t00;
	final Random ran = new Random();
	final static String datafile = "testTable.csv";
	private ExtendableTable h;
	private List<Object> colx;
	private List<Object> coly;
	private List<Object> colz;
	private List<Object> colf;
	
	@Before
	public void setUp() throws Exception {

		h = new ExtendableTable();
		h.addXYMembers();
		h.addNewMember("z");
		h.addNewMember("frame");
		h.addNewMember("roi");

		colx = h.getColumn("x");
		coly = h.getColumn("y");
		colz = h.getColumn("z");
		colf = h.getColumn("frame");
	}

	@Test
	public void test() {
		
		t0 = System.nanoTime();
		t00 = t0;
		double meanx = ran.nextDouble() * 10;
		double meany = ran.nextDouble() * 10;
		double meanz = ran.nextDouble() * 10;
		
		try {
			final File file = File.createTempFile("testTable", ".tmp", new File("/Users/ronny"));
			FileOutputStream os = new FileOutputStream(file);
			//ObjectOutputStream br = new ObjectOutputStream (new BufferedOutputStream(os));
			BufferedWriter br = new BufferedWriter(new OutputStreamWriter(os));
			br.write("x,y,z,frame\n");
		
			for (Integer i=0;i<N;i++){
				final double gx = ran.nextGaussian() + 5 + meanx;
				final double gy = ran.nextGaussian() + 5 + meany;
				final double gz = ran.nextGaussian() + 5 + meanz;
				colx.add( gx );
				coly.add( gy);
				colz.add( gz);
				colf.add( i+1);
				
				final String converted = gx + "," + gy + "," + gz + ","+ (i+1) +"\n";
				br.write(converted);
				long ct = System.nanoTime();
				long dt = ct - t0;
				
				t0=ct;
				
				sum+=dt;
				
				if (max < dt) {
					max=dt;
					id=i;
				}
				
				T[i] = dt;
				if (i % (N/10) == 0) 
					System.out.println(""+ (int)((float)i/N*100) +"%");
			}
			
//			br.writeInt(h.columnNames().size());
//			Iterator<String> cit = h.columnNames().iterator();
//			for (int k=0; k < h.columnNames().size();k++)
//				br.writeObject(cit.next());
//			Iterator<String> it = h.columnNames().iterator();
//			while (it.hasNext())
//				br.writeObject((FastTable<Object>) h.getColumn(it.next()));			
			
			br.close();
		} catch (IOException e){
			IJ.error(e.getMessage());
		}
		System.out.println("Total time is "+(System.nanoTime()-t00)/1e6);
		System.out.println("Average time is "+sum/N/1e6);
		System.out.println("Max time is "+max/1e6);		
		System.out.println("row is "+id);	
		System.gc();
	}

}
