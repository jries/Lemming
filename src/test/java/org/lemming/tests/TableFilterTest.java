package org.lemming.tests;

import static org.junit.Assert.assertNotEquals;

import java.io.File;
import org.junit.Before;
import org.junit.Test;
import org.lemming.modules.TableLoader;
import org.lemming.pipeline.ExtendableTable;

public class TableFilterTest {

	private ExtendableTable table;

	@Before
	public void setUp() throws Exception {
		
	}

	@Test
	public void test() {
		
		TableLoader loader = new TableLoader(new File("/home/ronny/ownCloud/storm/testTable.csv"));
		//loader.readObjects();
		loader.readCSV(',');
		table = loader.getTable();
		table.addFilterMinMax("x", 2, 7);
		long start=System.currentTimeMillis();
		ExtendableTable res = table.filter();
		System.out.println("original:"+table.getNumberOfRows()+" filtered:"+res.getNumberOfRows()
				+ " in " + (System.currentTimeMillis()-start) + "ms");
		assertNotEquals(table.getNumberOfRows(),res.getNumberOfRows());
	}

}
