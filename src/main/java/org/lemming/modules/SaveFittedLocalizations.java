package org.lemming.modules;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Locale;

import org.lemming.interfaces.Element;
import org.lemming.pipeline.FittedLocalization;
import org.lemming.pipeline.SingleRunModule;

public class SaveFittedLocalizations extends SingleRunModule {

	final private Locale curLocale;
	private File file;
	private FileWriter w;
	private long start;
	private int counter = 0;

	public SaveFittedLocalizations(File file) {
		this.curLocale = Locale.getDefault();
		this.file = file;
	}

	@Override
	public void beforeRun() {
		final Locale usLocale = new Locale("en", "US"); // setting us locale
		Locale.setDefault(usLocale);

		try {
			w = new FileWriter(file);
		} catch (IOException e) {
			e.printStackTrace();
		}

		start = System.currentTimeMillis();
	}

	@Override
	public Element process(Element data) {
		FittedLocalization loc = (FittedLocalization) data;
		if (loc == null)
			return null;
		if (loc.isLast()) {
			if (inputs.get(iterator).isEmpty()) {
				cancel();
				return null;
			}
			inputs.get(iterator).put(loc);
			return null;
		}

		StringBuilder out = new StringBuilder();

		out.append(loc.getID());
		out.append(", ");
		out.append(loc.getFrame());
		out.append(", ");
		out.append(loc.getX());
		out.append(", ");
		out.append(loc.getY());
		out.append(", ");
		out.append(loc.getZ());
		out.append(", ");
		out.append(loc.getsX());
		out.append(", ");
		out.append(loc.getsY());
		out.append("\n");

		try {
			w.write(out.toString());
		} catch (IOException e) {
			e.printStackTrace();
		}
		counter++;
		return null;
	}

	@Override
	public void afterRun() {
		try {
			w.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		Locale.setDefault(curLocale);
		System.out.println("" + counter + " fitted localizations saved in "
				+ (System.currentTimeMillis() - start) + "ms.");
	}

	@Override
	public boolean check() {
		// TODO Auto-generated method stub
		return false;
	}

}
