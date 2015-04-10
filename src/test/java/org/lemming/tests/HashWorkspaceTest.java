package org.lemming.tests;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;
import static org.lemming.interfaces.WorkspacePlugin.NEEDS_X;
import static org.lemming.interfaces.WorkspacePlugin.NEEDS_Y;
import static org.lemming.interfaces.WorkspacePlugin.NEEDS_Z;

import java.io.File;
//import java.io.FileWriter;
//import java.io.IOException;




import org.junit.Before;
import org.junit.Test;
import org.lemming.data.HashWorkspace;
import org.lemming.data.XYFLocalization;
import org.lemming.inputs.RandomLocalizer;
import org.lemming.interfaces.GenericLocalization;
import org.lemming.interfaces.GenericWorkspacePlugin;
import org.lemming.interfaces.IncompatibleWorkspaceException;
import org.lemming.interfaces.Localization;
import org.lemming.interfaces.Store;
import org.lemming.interfaces.WorkspacePlugin;
import org.lemming.outputs.GaussRenderOutput;
import org.lemming.outputs.PrintToFile;
import org.lemming.outputs.PrintToScreen;

/**
 * Test class for the HashWorkspace. 
 * 
 * @author Joe Borbely, Thomas Pengo
 */
public class HashWorkspaceTest extends HashWorkspace {

	@Before
	public void setUp() throws Exception {
	}

	@Test
	public void testAddRow() {
		HashWorkspace h = new HashWorkspace();
		
		h.addNewMember("x");
		h.addNewMember("y");
		
		GenericLocalization gi = h.newRow();
		gi.setX(1);
		gi.setY(1);
		
		assertEquals(h.getNumberOfRows(), 1);
	}

	@Test
	public void testCopy() {
		HashWorkspace h = new HashWorkspace();
		
		h.addNewMember("x");
		h.addNewMember("y");
		
		GenericLocalization gi = h.newRow();
		gi.setX(1);
		gi.setY(1);
		
		HashWorkspace h1 = new HashWorkspace();
		h1.addNewMember("x");
		h1.addNewMember("y");
		
		h1.addRow(h.getRow(0));
		
		assertEquals(h1.getRow(0).getX(), 1, 1e-3);
	}

	@Test
	public void testCopyAll() {
		HashWorkspace h = new HashWorkspace();
		
		h.addNewMember("x");
		h.addNewMember("y");
		
		GenericLocalization gi = h.newRow();
		gi.setX(1);
		gi.setY(1);
		
		HashWorkspace h1 = new HashWorkspace();
		h1.addNewMember("x");
		h1.addNewMember("y");
		
		h1.addAll(h);
		
		assertEquals(h1.getRow(0).getX(), 1, 1e-3);
	}

	@Test
	public void testCopyConstr() {
		HashWorkspace h = new HashWorkspace();
		
		h.addNewMember("x");
		h.addNewMember("y");
		
		GenericLocalization gi = h.newRow();
		gi.setX(1);
		gi.setY(1);
		
		HashWorkspace h1 = new HashWorkspace(h, true);
		assertEquals(h1.getRow(0).getX(), 1, 1e-3);
		
		HashWorkspace h2 = new HashWorkspace(h, false);
		GenericLocalization g = h2.newRow();
		g.setX(2);
		
		assertEquals(h2.getRow(0).getX(), 2, 1e-3);
	}


	@Test
	public void stressTest() {
			
		int N = (int) 10000000;
		
		HashWorkspace h = new HashWorkspace();
		
		h.addNewMember("x");
		h.addNewMember("y");
		h.addNewMember("z");
		h.addNewMember("frame");
		h.addNewMember("roi");
		
		
		long t0 = System.currentTimeMillis();
		
		long[] T = new long[N];
		
		long sum = 0;
		long max = 0;
		long id = 0;			
		
		
		for (int i=0;i<N;i++){				
			GenericLocalization gi = h.newRow();
			gi.setX(i);
			gi.setY(-i);
			gi.setZ(i);
			gi.set("roi", new double[] {1,2,3,4} );
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
		
		
		System.out.println("loop is "+ N);
		System.out.println("Average time is "+sum/N);
		System.out.println("Total time is "+sum);
		System.out.println("Max time is "+max);		
		System.out.println("row is "+id+":"+T[(int) id]);	
		System.gc();
	}
	
	@Test
	public void testRemove() {
		HashWorkspace h = new HashWorkspace();
		
		h.addNewMember("x");
		h.addNewMember("y");
		
		GenericLocalization gi = h.newRow();
		gi.setX(1);
		gi.setY(1);
		
		gi = h.newRow();
		gi.setX(2);
		gi.setY(2);		
		
		h.deleteRow(0);
		
		assertEquals(h.getRow(0).getX(), 2, 1e-3);
		assertEquals(h.getNumberOfRows(), 1);
	}

	@Test
	public void testFIFO() {
		HashWorkspace h = new HashWorkspace();
		
		h.addNewMember("x");
		h.addNewMember("y");
		h.addNewMember("id");
		
		GenericLocalization gi = h.newRow();
		gi.setX(1);
		gi.setY(1);
		gi.setID(1);
		
		gi = h.newRow();
		gi.setX(2);
		gi.setY(2);		
		gi.setID(2);		
		
		PrintToScreen ps = new PrintToScreen();
		ps.setInput(h.getFIFO());
		
		Thread t_print = new Thread(ps,"PrintToScreen");
		t_print.start();
		try {
			t_print.join(1000);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
	}

	
	@Test
	public void stressTestFIFO() {
		HashWorkspace h = new HashWorkspace();
		
		h.addNewMember("x");
		h.addNewMember("y");
		h.addNewMember("z");
		h.addNewMember("id"); // !!
		h.addNewMember("frame");
		h.addNewMember("roi");
		
		int N = (int) 1e6;
		long t0 = System.currentTimeMillis();
		
		for (int i=0; i<N; i++) {
			GenericLocalization gi = h.newRow();
			gi.setID(i);
			gi.setX(i);
			gi.setY(-i);
			gi.setZ(i);
			gi.set("roi", new double[] {1,2,3,4} );
		}
		
		System.out.println("Elapsed time "+ (System.currentTimeMillis()-t0));
		
		/*PrintToScreen ps = new PrintToScreen();
		ps.setInput(h.getFIFO());
		
		Thread t_print = new Thread(ps,"PrintToScreen");*/
		
		PrintToFile pf = new PrintToFile(new File("stressTestFIFO.csv"));
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

	@Test
	public void testPipeline() {
		HashWorkspace h = new HashWorkspace();
		h.setXname("x");
		h.setYname("y");
		
		Store<Localization> s = h.getFIFO();
		
		RandomLocalizer rl = new RandomLocalizer(50000, 256, 256);
		rl.setOutput(s);
		
		GaussRenderOutput gro = new GaussRenderOutput();
		gro.setInput(s);
		
		Thread t_rl = new Thread(rl,"RandomLocalizer");
		Thread t_gro = new Thread(rl,"GaussRenderOutput");
		
		t_rl.start();
		t_gro.start();
		try {
			t_rl.join();
			t_gro.join();
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		System.out.println("testPipeline finished");
	}
	
	@Test
	public void testToString() {
		HashWorkspace h = new HashWorkspace();
		h.setXname("x");
		h.setYname("y");
		
		Store<Localization> s = h.getFIFO();
		
		RandomLocalizer rl = new RandomLocalizer(1000, 256, 256);
		rl.setOutput(s);
		
		rl.run();
		
		System.out.println(h.toString());
		
	}
	
	@Test
	public void testIncompatibleWs() {
		HashWorkspace h = new HashWorkspace();
		h.setXname("x");
		h.setYname("y");
		
		Store<Localization> s = h.getFIFO();
		
		RandomLocalizer rl = new RandomLocalizer(1000, 256, 256);
		rl.setOutput(s);
		
		rl.run();
		
		WorkspacePlugin p = new GenericWorkspacePlugin();
		p.setRequiredMembers(NEEDS_X | NEEDS_Y | NEEDS_Z);
		try {
			p.setInput(h);
		} catch (IncompatibleWorkspaceException e) {
			// OK :)
		}		
	}
	
	@Test
	public void testCompatibleWs() {
		HashWorkspace h = new HashWorkspace();
		h.setXname("x");
		h.setYname("y");
		
		Store<Localization> s = h.getFIFO();
		
		RandomLocalizer rl = new RandomLocalizer(1000, 256, 256);
		rl.setOutput(s);
		
		rl.run();
		
		WorkspacePlugin p = new GenericWorkspacePlugin();
		p.setRequiredMembers(NEEDS_X | NEEDS_Y);
		try {
			p.setInput(h);
		} catch (IncompatibleWorkspaceException e) {
			fail();
		}		
	}
	
	@Test
	public void testAddLocalization() {
		HashWorkspace h = new HashWorkspace();

		Store<Localization> f = h.getFIFO();
		
		f.put(new XYFLocalization(1, 2, 3));
		
		boolean test_bool= h.hasMember("frame");
		
		assertEquals(test_bool, true);
		
		System.out.println(h.toString());
	}
	
}
