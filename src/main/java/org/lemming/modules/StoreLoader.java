package org.lemming.modules;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

import org.lemming.interfaces.Element;
import org.lemming.pipeline.MapElement;
import org.lemming.pipeline.SingleRunModule;

import ij.IJ;


public class StoreLoader extends SingleRunModule {
	
	private File file;
	private BufferedReader br;
	private String delimiter;
	private String sCurrentLine;
	private long start;
	private Locale curLocale;

	public StoreLoader(File f, String d){
		this.file = f;
		this.delimiter = d;
	}
	
	@Override
	public void beforeRun() {
		start = System.currentTimeMillis();
		iterator = outputs.keySet().iterator().next();
		
		curLocale = Locale.getDefault();
		final Locale usLocale = new Locale("en", "US"); // setting us locale
		Locale.setDefault(usLocale);
		
		try {
			br = new BufferedReader(new FileReader(file));	
			sCurrentLine = br.readLine();	
			if (sCurrentLine==null) throw new NullPointerException("first line is null!");
		} catch (IOException e) {
			IJ.error(e.getMessage());
		}
	}

	@Override
	public Element process(Element data) { // data not used here

		String[] s = sCurrentLine.split(delimiter);
		Map<String,Object> row = new HashMap<>(s.length);
		
		for (int i = 0; i < s.length; i++){
			String c = s[i].trim();
			if (c.split(".").length >1)
				row.put("col"+i, Double.parseDouble(c));
			else
				row.put("col"+i, Integer.parseInt(c));
		}
							
		MapElement me = new MapElement(row);
		
		try {
			sCurrentLine = br.readLine();
		} catch (IOException e1) {
			IJ.error(e1.getMessage());
		}
		if (sCurrentLine == null) {
			me.setLast(true);
			cancel();
		}
		return me;
	}
	
	@Override
	public void afterRun() {
		try {
			br.close();
		} catch (IOException e) {
			IJ.error(e.getMessage());
		}
		Locale.setDefault(curLocale);
		System.out.println("Reading data done in " + (System.currentTimeMillis()-start) + "ms.");
	}

	@Override
	public boolean check() {
		return outputs.size()>=1;
	}

}
