package org.lemming.tests;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;
import org.lemming.data.HashWorkspace;
import org.lemming.input.WorkspaceRandomLocalizer;

public class WorkspaceRandomLocalizerTest {

	@Before
	public void setUp() throws Exception {
	}

	@Test
	public void test() {
		HashWorkspace h = new HashWorkspace();
		
		WorkspaceRandomLocalizer w = new WorkspaceRandomLocalizer(100,256,256);
		w.setOutput(h);
		w.run();
		
		System.out.println(h);
		
		assertEquals(h.getNumberOfRows(), 100);
	}

}
