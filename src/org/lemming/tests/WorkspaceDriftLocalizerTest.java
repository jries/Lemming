package org.lemming.tests;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;
import org.lemming.data.HashWorkspace;
import org.lemming.inputs.WorkspaceDriftLocalizer;
import org.lemming.inputs.WorkspaceRandomLocalizer;

/**
 * Test class for creating randomly-positioned localizations with drift and 
 * adding the localizations into a HashWorkspace.
 * 
 * @author Joe Borbely, Thomas Pengo, Joran Deschamps
 */
public class WorkspaceDriftLocalizerTest {

	HashWorkspace h;
	WorkspaceDriftLocalizer w;
	
	@Before
	public void setUp() throws Exception {
		h = new HashWorkspace();		
		w = new WorkspaceDriftLocalizer(100);
	}

	@Test
	public void test() {
		w.setOutput(h);
		w.run();
		
		//System.out.println(h);
		
		assertTrue(h.hasMember("id"));
		assertTrue(h.hasMember("x"));
		assertTrue(h.hasMember("y"));
		assertTrue(h.hasMember("frame")); 
		assertEquals(h.getNumberOfRows(), 100);
	}

}
