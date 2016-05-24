package org.lemming.modules;

import java.io.File;
import java.io.IOException;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.StringTokenizer;

import org.lemming.interfaces.Element;
import org.lemming.pipeline.ElementMap;
import org.lemming.pipeline.SingleRunModule;

import ij.IJ;
import org.lemming.tools.CharBufferedReader;

/**
 * loading localizations onto a queue
 * 
 * @author Ronny Sczech
 *
 */
public class StoreLoader extends SingleRunModule {
	
	private final File file;
	private CharBufferedReader br;
	private final String delimiter;
	private String sCurrentLine;
	private Locale curLocale;
	private final Map<String,Object> metaData = new HashMap<>();
	private final Set<String> header = new LinkedHashSet<>();
	private String[] nameArray = null;

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
			br = new CharBufferedReader(new InputStreamReader(new FileInputStream(file)));
			sCurrentLine = br.readLine();	
			if (sCurrentLine==null) throw new NullPointerException("first line is null!");
			while (sCurrentLine.startsWith("#")){
				sCurrentLine = br.readLine();
				addMetadata(sCurrentLine);
			}
			final StringTokenizer s = new StringTokenizer(sCurrentLine,String.valueOf(delimiter));
			while(s.hasMoreTokens())
				header.add(s.nextToken().trim());
			nameArray  = header.toArray(new String[]{});	
			sCurrentLine = br.readLine();
		} catch (IOException e) {
			IJ.error(e.getMessage());
		}
	}

	@Override
	public Element processData(Element data) { // data not used here

		final String[] s = sCurrentLine.split(delimiter);
		final Map<String,Number> row = new HashMap<>(s.length);
		
		for (int i = 0; i < s.length; i++){
			String c = s[i].trim();
			if (c.contains("."))
				row.put(nameArray[i], Double.parseDouble(c));
			else
				row.put(nameArray[i], Integer.parseInt(c));
		}
							
		final ElementMap me = new ElementMap(row);
		
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
	
	private void addMetadata(String hashedline) {
		final String line = hashedline.substring(1);
		final String[] parsed = line.split("=");
		if (parsed.length == 2)
			metaData.put(parsed[0].trim(), parsed[1].trim());
	}
	
	public Map<String, Object> getMetadata(){
		return metaData;
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
