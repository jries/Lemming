package org.lemming.tests;

import static org.junit.Assert.*;

import org.junit.Test;
import org.lemming.*;
import org.lemming.data.GenericLocalization;
import org.lemming.data.HashWorkspace;
import org.lemming.outputs.PrintToScreen;
import org.lemming.utils.SortWorkspace;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
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
		int N = (int) 	1e7;
		Random rnd = new Random();
		for (int i=0;i<N;i++){
			GenericLocalization gi = h.newRow();
			gi.setX(rnd.nextFloat());
			gi.setY(rnd.nextFloat());
			gi.setFrame(rnd.nextInt(N));
			gi.setChannel(rnd.nextInt(2));
			gi.setID(i);
			
		}
		
		String[] membersList = {"x","frame"};
		long t0 = System.currentTimeMillis();
		SortWorkspace sorter=new SortWorkspace(h,membersList);
		long t1 = System.currentTimeMillis()-t0;
		//List Frames=h.getMember("frame");
		HashWorkspace ws2=sorter.sortedWorkspace();
		h=ws2;
		long t2 = System.currentTimeMillis()-t1-t0;
		//List Frames2=h.getMember("frame");
		
		System.out.println("sort "+t1);
		System.out.println("rearrange "+t2);
		
		PrintToScreen ps = new PrintToScreen();
		ps.setInput(h.getFIFO());
		
		//ps.run();

	}

}
