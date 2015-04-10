package org.lemming.tests;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;
import org.lemming.data.HashWorkspace;
import org.lemming.inputs.WorkspaceRandomLocalizer;

/**
 * Test class for creating randomly-positioned localizations and 
 * adding the localizations into a HashWorkspace.
 * 
 * @author Joe Borbely, Thomas Pengo
 */
public class WorkspaceRandomLocalizerTest {

	HashWorkspace h;
	WorkspaceRandomLocalizer w;
	
	@Before
	public void setUp() throws Exception {
		h = new HashWorkspace();		
		w = new WorkspaceRandomLocalizer(100, 256, 256);
	}

	@Test
	public void test() {
		w.setOutput(h);
		w.run();
		
		System.out.println(h.toString());
		
		assertTrue(h.hasMember("id"));
		assertTrue(h.hasMember("x"));
		assertTrue(h.hasMember("y"));
		assertTrue(h.hasMember("y")); 
		assertEquals(h.getNumberOfRows(), 100);
	}

}
