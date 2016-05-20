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
	private final int N = (int) 1e6;
	private final long[] T = new long[N];
	private long sum = 0;
	private long max = 0;
	private long id = 0;
	private long t0;
	private long t00;
	private final Random ran = new Random();
	private ExtendableTable h;


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
		
		bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(System.getProperty("user.home")+"/ownCloud/geomTable.csv")));
	}

	@Test
	public void test() {
		
		double meanx = 5;
		double meany = 5;
		double meanz = 5;
		double r = 3;
		
		try {
			bw.write("x,y,z,frame,sx,sy\n");
			List<Number> colx = h.getColumn("x");
			List<Number> coly = h.getColumn("y");
			List<Number> colz = h.getColumn("z");
			List<Number> colf = h.getColumn("frame");
			List<Number> colsX = h.getColumn("sX");
			List<Number> colsY = h.getColumn("sY");
		
			for (Integer i=0;i<N;i++){
				final double gx = r + Math.cos(ran.nextDouble()*2*Math.PI)*meanx;
				final double gy = r + Math.sin(ran.nextDouble()*2*Math.PI)*meany;
				final double gz = r * meanz;
				final double sX = Math.abs(ran.nextGaussian()/10);
				final double sY = Math.abs(ran.nextGaussian()/10);
				colx.add( gx);
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
