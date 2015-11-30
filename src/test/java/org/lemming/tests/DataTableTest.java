package org.lemming.tests;

import static org.junit.Assert.*;

import java.io.File;
import java.util.Map;

import org.junit.Before;
import org.junit.Test;
import org.lemming.interfaces.Store;
import org.lemming.modules.DataTable;
import org.lemming.modules.ReadLocalizationPrecision3D;
import org.lemming.pipeline.ExtendableTable;
import org.lemming.pipeline.Manager;

public class DataTableTest {
	
	private Manager pipe;
	private ExtendableTable table;
	private Map<Integer, Store> map;

	@Before
	public void setUp() throws Exception {
		pipe = new Manager();
		
		ReadLocalizationPrecision3D reader = new ReadLocalizationPrecision3D(new File("/home/ronny/Bilder/fitted.csv"),",");
		pipe.add(reader);
		
		DataTable workspace = new DataTable();
		table = workspace.getTable();
		pipe.add(workspace);
		
		pipe.linkModules(reader, workspace, true, 128);
		map = pipe.getMap();
	}

	@Test
	public void test() {
		pipe.run();
		System.out.println("Table rows: " + table.getNumberOfRows());
		assertEquals(true,map.values().iterator().next().isEmpty());
	}

}
