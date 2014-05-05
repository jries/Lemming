package org.lemming.tests;

import static org.junit.Assert.*;

import org.junit.Test;
import org.lemming.*;
import org.lemming.data.GenericLocalization;
import org.lemming.data.HashWorkspace;
import org.lemming.outputs.PrintToScreen;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

/**
 * Test class for the sort method of HashWorkspace. 
 * 
 * @author Jonas Ries
 */
public class SortHashWorkspaceTest {

	@Test
	public void test() {
	HashWorkspace h = new HashWorkspace();	
		h.addNewMember("x");
		h.addNewMember("y");
		h.addNewMember("z");
		h.addNewMember("id");
		h.addNewMember("frame");
		h.addNewMember("channel");
		int N = (int) 2e6;
		Random rnd = new Random();
		for (int i=0;i<N;i++){
			GenericLocalization gi = h.newRow();
			gi.setX(rnd.nextFloat());
			gi.setY(rnd.nextFloat());
			gi.setFrame(rnd.nextInt(N/10));
			gi.setChannel(rnd.nextInt(2));
			gi.setID(i);
			
		}
		String[] membersList = {"x","frame"};
		long t0 = System.currentTimeMillis();
		h.sortMembers(membersList);
		long ct = System.currentTimeMillis();
		long dt=ct-t0;
		System.out.println(dt);
		PrintToScreen ps = new PrintToScreen();
		ps.setInput(h.getFIFO());
		
		//ps.run();

	}

}
