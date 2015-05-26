package org.lemming.modules;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import org.lemming.pipeline.Element;
import org.lemming.pipeline.Localization;
import org.lemming.pipeline.SingleRunModule;


public class ReadLocalizations extends SingleRunModule {
	
	private File file;
	private BufferedReader br;
	private String delimiter;
	private String sCurrentLine;
	private long start;

	public ReadLocalizations(File f, String d){
		this.file = f;
		this.delimiter = d;
	}
	
	@Override
	public void beforeRun() {
		start = System.currentTimeMillis();
		iterator = outputs.keySet().iterator().next();
		
		try {
			br = new BufferedReader(new FileReader(file));	
			sCurrentLine = br.readLine();	
			if (sCurrentLine==null) throw new NullPointerException("first line is null!");
		} catch (IOException e) {
			System.err.println(e.getMessage());
		}
	}

	@Override
	public void process(Element data) { // data not used here
		try {
			String[] s = sCurrentLine.split(delimiter);
			for (int i = 0; i<s.length;i++)
				s[i]= s[i].trim();
			if (s.length > 3) {
				Localization localization =  new Localization(Long.parseLong(s[1]), Double.parseDouble(s[2]), Double.parseDouble(s[3]));
				
				sCurrentLine = br.readLine();
				if (sCurrentLine == null){
					cancel();
					localization.setLast(true);
				}
				outputs.get(iterator).put(localization);				
			}
		} catch (IOException e1) {
			System.err.println(e1.getMessage());
		}
	}
	
	@Override
	public void afterRun() {
		try {
			br.close();
		} catch (IOException e) {
			System.err.println(e.getMessage());
		}
		System.out.println("Reading of localizations done in " + (System.currentTimeMillis()-start) + "ms.");
	}

}
