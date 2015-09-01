package org.lemming.tests;

import static org.junit.Assert.*;

import java.io.File;

import org.junit.Before;
import org.junit.Test;
import org.lemming.modules.DataTable;
import org.lemming.modules.ReadFittedLocalizations;
import org.lemming.pipeline.ExtendableTable;
import org.lemming.pipeline.FastStore;
import org.lemming.pipeline.Pipeline;

public class DataTableTest {
	
	private Pipeline pipe;
	private FastStore locs;
	private ExtendableTable table;

	@Before
	public void setUp() throws Exception {
		pipe = new Pipeline("test");
		locs = new FastStore();
		
		ReadFittedLocalizations reader = new ReadFittedLocalizations(new File("/home/ronny/Bilder/fitted.csv"),",");
		reader.setOutput(locs);
		pipe.add(reader);
		
		DataTable workspace = new DataTable();
		workspace.setInput(locs);
		table = workspace.getTable();
		pipe.add(workspace);
	}

	@Test
	public void test() {
		pipe.run();
		System.out.println("Table rows: " + table.getNumberOfRows());
		assertEquals(true,locs.isEmpty());
	}

}
