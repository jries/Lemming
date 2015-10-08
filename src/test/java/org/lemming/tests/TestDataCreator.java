package org.lemming.tests;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.List;
import java.util.Locale;
import java.util.Random;

import org.junit.Before;
import org.junit.Test;
import org.lemming.pipeline.ExtendableTable;

import ij.IJ;

public class TestDataCreator {
	private BufferedWriter bw;
	final int N = (int) 1e6;
	final long[] T = new long[N];
	long sum = 0;
	long max = 0;
	long id = 0;
	long t0,t00;
	final Random ran = new Random();
	final static String datafile = "testTable.csv";
	private ExtendableTable h;
	private List<Number> colx;
	private List<Number> coly;
	private List<Number> colz;
	private List<Number> colsX;
	private List<Number> colsY;
	private List<Number> colf;

	
	@Before
	public void setUp() throws Exception {
		t0 = System.nanoTime();
		t00 = t0;
		
		h = new ExtendableTable();
		h.addXYMembers();
		h.addNewMember("z");
		h.addNewMember("frame");
		h.addNewMember("sX");
		h.addNewMember("sY");
		
		bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("/Users/ronny/ownCloud/storm/testTable.csv")));
	}

	@Test
	public void test() {
		
		double meanx = ran.nextDouble() * 10;
		double meany = ran.nextDouble() * 10;
		double meanz = ran.nextDouble() * 10;
		
		try {
			bw.write("x,y,z,frame,sx,sy\n");
			colx = h.getColumn("x");
			coly = h.getColumn("y");
			colz = h.getColumn("z");
			colf = h.getColumn("frame");
			colsX = h.getColumn("sX");
			colsY = h.getColumn("sY");
		
			for (Integer i=0;i<N;i++){
				final double gx = ran.nextGaussian() + 5 + meanx;
				final double gy = ran.nextGaussian() + 5 + meany;
				final double gz = ran.nextGaussian() + 5 + meanz;
				final double sX = Math.abs(ran.nextGaussian());
				final double sY = Math.abs(ran.nextGaussian());
				colx.add( gx );
				coly.add( gy);
				colz.add( gz);
				colf.add( i+1);
				colsX.add(sX);
				colsY.add(sY);
				
				final String converted = String.format(Locale.US, "%.14f,%.14f,%.14f,%d,%.14f,%.14f\n",gx,gy,gz,i+1,sX,sY);
				bw.write(converted);
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
			
			bw.close();
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
