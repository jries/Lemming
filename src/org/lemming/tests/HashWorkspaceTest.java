package org.lemming.tests;

import static org.junit.Assert.*;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;

import org.junit.Before;
import org.junit.Test;
import org.lemming.data.GenericLocalization;
import org.lemming.data.HashWorkspace;
import org.lemming.data.Localization;
import org.lemming.data.Store;
import org.lemming.data.Workspace;
import org.lemming.input.RandomLocalizer;
import org.lemming.interfaces.GenericWorkspacePlugin;
import org.lemming.interfaces.IncompatibleWorkspaceException;
import org.lemming.interfaces.WorkspacePlugin;
import org.lemming.outputs.GaussRenderOutput;
import org.lemming.outputs.PrintToFile;
import org.lemming.outputs.PrintToScreen;

import static org.lemming.interfaces.GenericWorkspacePlugin.*;

public class HashWorkspaceTest extends HashWorkspace {

	@Before
	public void setUp() throws Exception {
	}

	@Test
	public void test() {
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
		HashWorkspace h = new HashWorkspace();
		
		h.addNewMember("x");
		h.addNewMember("y");
		h.addNewMember("z");
		h.addNewMember("frame");
		h.addNewMember("roi");
		
		int N = (int) 1e6;
		long t0 = System.currentTimeMillis();
		
		long[] T = new long[N];
		
		double sum = 0;
		double max = 0;
		for (int i=0; i<N; i++) {
			GenericLocalization gi = h.newRow();
			gi.setX(i);
			gi.setY(-i);
			gi.setZ(i);
			gi.set("roi", new double[] {1,2,3,4} );
			
			long ct = System.currentTimeMillis();
			long dt = ct - t0;
			t0 = ct;
			
			sum+=dt;
			
			if (max < dt) {
				max = dt;
			}
			
			T[i] = dt;
		}
		
		System.out.println("Average time is "+sum/N);
		System.out.println("Total time is "+sum);
		System.out.println("Max time is "+max);
		
		try {
			FileWriter fw = new FileWriter("out.csv");
			for (long l : T) fw.write(String.format("%d\n", l));
			fw.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
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
		
		ps.run();
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
		
		long[] T = new long[N];
		
		double sum = 0;
		double max = 0;
		for (int i=0; i<N; i++) {
			GenericLocalization gi = h.newRow();
			gi.setID(i);
			gi.setX(i);
			gi.setY(-i);
			gi.setZ(i);
			gi.set("roi", new double[] {1,2,3,4} );
			
			long ct = System.currentTimeMillis();
			long dt = ct - t0;
			t0 = ct;
			
			sum+=dt;
			
			if (max < dt) {
				max = dt;
			}
			
			T[i] = dt;
		}
		
		System.out.println("Average time is "+sum/N);
		System.out.println("Total time is "+sum);
		System.out.println("Max time is "+max);
		
		PrintToScreen ps = new PrintToScreen();
		ps.setInput(h.getFIFO());
		
		ps.run();

		PrintToFile pf = new PrintToFile(new File("stressTestFIFO.csv"));
		pf.setInput(h.getFIFO());
		
		pf.run();
	}

	@Test
	public void testPipeline() {
		HashWorkspace h = new HashWorkspace();
		h.setXname("x");
		h.setYname("y");
		
		Store<Localization> s = h.getFIFO();
		
		RandomLocalizer rl = new RandomLocalizer(50000, 256, 256);
		rl.setOutput(s);
		
		rl.run();

		GaussRenderOutput gro = new GaussRenderOutput();
		gro.setInput(s);
		
		gro.run();
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
	
}
